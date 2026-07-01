import os
import pandas as pd
import numpy as np
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from ml_pipeline import clean_data, feature_engineering, preprocess_data, evaluate_model, load_data
from shap_explainer import SHAPExplainer
import joblib
import json
from dotenv import load_dotenv
from docx import Document

# Load environment variables
load_dotenv()

# ==========================================
# 1. Define State
# ==========================================
class AgentState(TypedDict):
    csv_path: str
    raw_data: pd.DataFrame
    processed_data: pd.DataFrame
    preprocessed_x: pd.DataFrame
    employee_ids: List[str]
    predictions: List[Dict[str, Any]]
    recommendations: List[Dict[str, str]]
    error: str

# ==========================================
# 2. Define Agent Nodes
# ==========================================

def data_ingestion_agent(state: AgentState):
    """Loads and validates the CSV data."""
    print("Agent: Data Ingestion")
    try:
        df = load_data(state["csv_path"])
        df_clean = clean_data(df)
        return {"raw_data": df, "processed_data": df_clean}
    except Exception as e:
        return {"error": str(e)}

def feature_engineering_agent(state: AgentState):
    """Engineers predictive features."""
    print("Agent: Feature Engineering")
    try:
        df_clean = state["processed_data"]
        # Save employee IDs before dropping
        emp_ids = df_clean['Employee_ID'].tolist()
        
        df_features = feature_engineering(df_clean)
        
        # Load preprocessing artifacts
        scaler = joblib.load('scaler.pkl')
        with open('preprocessing_info.json', 'r') as f:
            info = json.load(f)
            
        # We need to preprocess for inference
        X = df_features.drop('Attrition', axis=1, errors='ignore')
        cat_cols = info['cat_cols']
        num_cols = info['num_cols']
        
        # In a real scenario we'd use saved LabelEncoders, but for simplicity here we'll just pandas dummy or re-encode
        # Since this is a prototype, we'll re-run a simplified preprocess that works with the scaler
        # We need to ensure columns match exactly what the model expects.
        # For inference robustness, this usually requires a robust ColumnTransformer. 
        # For now, we will do a direct transform assuming data shape is identical to train.
        
        from sklearn.preprocessing import LabelEncoder
        for col in cat_cols:
            le = LabelEncoder()
            le.classes_ = np.array(info['label_encoders_classes'][col])
            # Handle unseen labels by assigning to '0' or unknown. 
            # Simplified: assuming all labels were seen in dummy data
            X[col] = le.transform(X[col])
            
        X[num_cols] = scaler.transform(X[num_cols])
        
        return {"preprocessed_x": X, "employee_ids": emp_ids}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def prediction_and_scoring_agent(state: AgentState):
    """Scores employees using the trained XGBoost model and explains predictions with SHAP."""
    print("Agent: Prediction & Scoring")
    try:
        X = state["preprocessed_x"]
        emp_ids = state["employee_ids"]
        
        model = joblib.load('attrition_xgboost_model.pkl')
        explainer = SHAPExplainer('attrition_xgboost_model.pkl')
        
        # Get probabilities
        probs = model.predict_proba(X)[:, 1]
        
        with open('preprocessing_info.json', 'r') as f:
            info = json.load(f)
        feature_names = info['cat_cols'] + info['num_cols']
        
        predictions = []
        # Score and explain only High Risk employees to save time/API calls
        for i in range(len(probs)):
            prob = float(probs[i])
            if prob > 0.60: # Threshold for high risk
                tier = "Critical" if prob > 0.80 else "High"
                
                # Get SHAP explanation
                x_instance = pd.DataFrame([X.iloc[i]], columns=feature_names)
                explanation = explainer.explain_prediction(feature_names, x_instance)
                
                predictions.append({
                    "Employee_ID": emp_ids[i],
                    "Risk_Probability": round(prob * 100, 2),
                    "Risk_Tier": tier,
                    "Top_Drivers": explanation["top_drivers"],
                    "SHAP_Base_Value": explanation["base_value"],
                    "SHAP_All_Impacts": explanation["all_impacts"]
                })
        
        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}

def recommendation_agent(state: AgentState):
    """Uses Llama 3 via Groq to draft retention recommendations."""
    print("Agent: Recommendation Drafting")
    try:
        predictions = state.get("predictions", [])
        if not predictions:
            return {"recommendations": []}
            
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
        
        prompt_template = PromptTemplate(
            input_variables=["emp_id", "risk", "drivers"],
            template="""
            You are an expert HR consultant. We have an employee (ID: {emp_id}) who has a {risk}% probability of leaving the company.
            Based on our ML analysis, the top factors driving their attrition risk are:
            {drivers}
            
            Write a short, highly professional, personalized 3-point action plan for their manager to help retain this employee.
            Be specific to the driving factors provided. Output ONLY the action plan, no pleasantries.
            """
        )
        
        recommendations = []
        for pred in predictions:
            drivers_str = "\\n".join([f"- {d['feature']} ({d['impact_direction']})" for d in pred['Top_Drivers']])
            prompt = prompt_template.format(
                emp_id=pred["Employee_ID"],
                risk=pred["Risk_Probability"],
                drivers=drivers_str
            )
            response = llm.invoke(prompt)
            
            recommendations.append({
                "Employee_ID": pred["Employee_ID"],
                "Action_Plan": response.content
            })
            
        return {"recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}

def report_generation_agent(state: AgentState):
    """Generates Word documents for the HR team."""
    print("Agent: Report Generation")
    try:
        recommendations = state.get("recommendations", [])
        
        if not os.path.exists("reports"):
            os.makedirs("reports")
            
        for rec in recommendations:
            doc = Document()
            doc.add_heading(f"Retention Action Plan: {rec['Employee_ID']}", 0)
            doc.add_paragraph("AI-Generated Retention Strategy based on Attrition Risk Drivers.")
            doc.add_paragraph(rec['Action_Plan'])
            doc.save(f"reports/ActionPlan_{rec['Employee_ID']}.docx")
            
        return state
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 3. Build Graph
# ==========================================
def build_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("data_ingestion", data_ingestion_agent)
    workflow.add_node("feature_engineering", feature_engineering_agent)
    workflow.add_node("prediction_scoring", prediction_and_scoring_agent)
    workflow.add_node("recommendation", recommendation_agent)
    workflow.add_node("report_generation", report_generation_agent)
    
    # Define edges (The flow)
    workflow.add_edge("data_ingestion", "feature_engineering")
    workflow.add_edge("feature_engineering", "prediction_scoring")
    workflow.add_edge("prediction_scoring", "recommendation")
    workflow.add_edge("recommendation", "report_generation")
    workflow.add_edge("report_generation", END)
    
    # Set entry point
    workflow.set_entry_point("data_ingestion")
    
    # Compile
    return workflow.compile()

if __name__ == "__main__":
    # Test the graph
    import numpy as np
    print("Testing LangGraph Workflow...")
    app = build_agent_graph()
    
    initial_state = {
        "csv_path": "mock_hr_data.csv"
    }
    
    # Run the graph
    result = app.invoke(initial_state)
    
    if "error" in result:
        print(f"Error occurred: {result['error']}")
    else:
        print("\\n--- Workflow Complete ---")
        print(f"Total High Risk Employees Found: {len(result.get('predictions', []))}")
        print(f"Action Plans Generated: {len(result.get('recommendations', []))}")
        if result.get('recommendations'):
            print("\\nSample Recommendation:")
            print(result['recommendations'][0]['Action_Plan'])
