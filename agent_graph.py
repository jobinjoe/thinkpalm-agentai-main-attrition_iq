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
        
        # Document the transformation
        import os
        os.makedirs("reports", exist_ok=True)
        with open("reports/feature_engineering_log.txt", "w") as f:
            f.write(f"Feature Engineering Complete.\\n")
            f.write(f"Original Rows: {len(df_clean)}, New Features Generated: {len(df_features.columns) - len(df_clean.columns)}\\n")
            f.write(f"Current Features: {list(df_features.columns)}\\n")
            
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
        # Score and explain all employees for the dashboard
        for i in range(len(probs)):
            prob = float(probs[i])
            
            if prob > 0.80:
                tier = "Critical"
            elif prob > 0.60:
                tier = "High"
            elif prob > 0.30:
                tier = "Medium"
            else:
                tier = "Low"
                
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
            
            First, write a single plain English sentence summarizing these top attrition drivers (e.g., "This employee's top attrition drivers are: working hours (increased), no promotion (increased), below-band salary (increased).").
            
            Then, write a short, highly professional, personalized 3-point action plan for their manager to help retain this employee.
            Be specific to the driving factors provided. Output ONLY the summary sentence followed by the action plan, no pleasantries.
            """
        )
        
        recommendations = []
        for pred in predictions:
            if pred.get("Risk_Tier") not in ["High", "Critical"]:
                continue
                
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
    """Generates Word documents and a full HTML Report for the HR team."""
    print("Agent: Report Generation")
    try:
        recommendations = state.get("recommendations", [])
        predictions = state.get("predictions", [])
        
        if not os.path.exists("reports"):
            os.makedirs("reports")
            
        # 1. Generate Individual Word Docs
        for rec in recommendations:
            doc = Document()
            doc.add_heading(f"Retention Action Plan: {rec['Employee_ID']}", 0)
            doc.add_paragraph("AI-Generated Retention Strategy based on Attrition Risk Drivers.")
            doc.add_paragraph(rec['Action_Plan'])
            doc.save(f"reports/ActionPlan_{rec['Employee_ID']}.docx")
            
        # 2. Generate Full HTML Report
        high_risk = [p for p in predictions if p.get("Risk_Tier") in ["High", "Critical"]]
        
        html_content = f"""
        <html>
        <head><title>Attrition Risk Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .critical {{ color: #e74c3c; font-weight: bold; }}
            .high {{ color: #e67e22; font-weight: bold; }}
        </style>
        </head>
        <body>
            <h1>Weekly Attrition Risk Report</h1>
            <p>Total High/Critical Risk Employees Found: <strong>{len(high_risk)}</strong></p>
            <h2>Top At-Risk Employees</h2>
            <table>
                <tr>
                    <th>Employee ID</th>
                    <th>Risk Probability</th>
                    <th>Risk Tier</th>
                </tr>
        """
        
        for p in sorted(high_risk, key=lambda x: x["Risk_Probability"], reverse=True)[:10]:
            tier_class = "critical" if p["Risk_Tier"] == "Critical" else "high"
            html_content += f"""
                <tr>
                    <td>{p['Employee_ID']}</td>
                    <td>{p['Risk_Probability']}%</td>
                    <td class="{tier_class}">{p['Risk_Tier']}</td>
                </tr>
            """
            
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open("reports/Attrition_Risk_Report.html", "w") as f:
            f.write(html_content)
            
        return state
    except Exception as e:
        return {"error": str(e)}

def alert_agent(state: AgentState):
    """Sends Email and Slack alerts."""
    print("Agent: Alert")
    try:
        import smtplib
        import requests
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        predictions = state.get("predictions", [])
        high_risk_count = len([p for p in predictions if p.get("Risk_Tier") in ["High", "Critical"]])
        
        # SLACK WEBHOOK
        slack_url = os.environ.get("SLACK_WEBHOOK_URL")
        if slack_url:
            payload = {"text": f"🚨 *AttritionIQ Alert*: Analysis complete. Found {high_risk_count} high-risk employees. Check the dashboard!"}
            requests.post(slack_url, json=payload)
        else:
            print("Skipping Slack alert (SLACK_WEBHOOK_URL not set).")
            
        # SMTP EMAIL
        smtp_server = os.environ.get("SMTP_SERVER")
        smtp_port = os.environ.get("SMTP_PORT")
        smtp_user = os.environ.get("SMTP_USER")
        smtp_pass = os.environ.get("SMTP_PASS")
        
        if smtp_server and smtp_user and smtp_pass:
            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = "thinkhr@yopmail.com"  # Send to specific HR email
            msg['Subject'] = "Weekly Attrition Risk Report"
            
            body = f"Analysis complete. Found {high_risk_count} high-risk employees. See the attached HTML report."
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach HTML report if exists
            report_path = "reports/Attrition_Risk_Report.html"
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    attachment = MIMEText(f.read(), "html")
                    attachment.add_header('Content-Disposition', 'attachment', filename="Attrition_Risk_Report.html")
                    msg.attach(attachment)
            
            with smtplib.SMTP(smtp_server, int(smtp_port or 587)) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        else:
            print("Skipping SMTP email (SMTP credentials not fully set).")
            
        return state
    except Exception as e:
        import traceback
        traceback.print_exc()
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
    workflow.add_node("alert", alert_agent)
    
    # Define edges (The flow)
    workflow.add_edge("data_ingestion", "feature_engineering")
    workflow.add_edge("feature_engineering", "prediction_scoring")
    workflow.add_edge("prediction_scoring", "recommendation")
    workflow.add_edge("recommendation", "report_generation")
    workflow.add_edge("report_generation", "alert")
    workflow.add_edge("alert", END)
    
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
