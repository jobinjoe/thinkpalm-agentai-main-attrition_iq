import streamlit as st
import pandas as pd
import plotly.express as px
import os
from agent_graph import build_agent_graph

st.set_page_config(page_title="AttritionIQ Dashboard", page_icon="📈", layout="wide")

st.title("📈 AttritionIQ: Agentic Attrition Prediction Platform")
st.markdown("Proactively identify and retain your top talent using AI Agents.")

# Initialize session state
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None

# Sidebar for Upload
with st.sidebar:
    st.header("1. Upload HR Data")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        # Save temp file
        with open("temp_upload.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("File uploaded successfully!")
        
        if st.button("Run AI Analysis", type="primary"):
            # Immediate API Key validation
            api_key_valid = True
            if not os.environ.get("GROQ_API_KEY"):
                st.error("Error: GROQ_API_KEY is not set. Please add your Groq API key to the .env file.")
                api_key_valid = False
            else:
                try:
                    from langchain_groq import ChatGroq
                    test_llm = ChatGroq(model="llama-3.1-8b-instant", max_tokens=1)
                    test_llm.invoke("test")
                except Exception as e:
                    st.error(f"API Key Validation Failed: {str(e)}")
                    api_key_valid = False
            
            if api_key_valid:
                with st.spinner("Agents are analyzing data, training models, and writing retention plans..."):
                    app = build_agent_graph()
                    result = app.invoke({"csv_path": "temp_upload.csv"})
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.session_state.predictions = result.get("predictions", [])
                        st.session_state.recommendations = result.get("recommendations", [])
                        st.session_state.raw_data = result.get("raw_data")
                        st.success("Analysis Complete!")

# Main Content
if st.session_state.raw_data is not None and st.session_state.predictions:
    tab1, tab2, tab3 = st.tabs(["📊 Organization Overview", "👤 Employee Drill-Down", "📉 Model Metrics"])
    
    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.header("Organization Attrition Risk Heatmap")
        
        # Merge predictions with raw data for department info
        preds_df = pd.DataFrame(st.session_state.predictions)
        merged_df = pd.merge(st.session_state.raw_data, preds_df, on="Employee_ID")
        
        # Aggregate by Department
        dept_risk = merged_df.groupby("Department")["Risk_Probability"].mean().reset_index()
        dept_risk = dept_risk.sort_values(by="Risk_Probability", ascending=False)
        
        fig = px.bar(dept_risk, x="Department", y="Risk_Probability", color="Risk_Probability",
                     color_continuous_scale="Reds", title="Average Attrition Risk by Department (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Critical Risk Employees")
        critical_df = preds_df[preds_df['Risk_Tier'] == 'Critical'].copy()
        if not critical_df.empty:
            st.dataframe(critical_df[['Employee_ID', 'Risk_Probability', 'Risk_Tier']], use_container_width=True)
        else:
            st.info("No critical risk employees found.")

    # --- TAB 2: DRILL DOWN ---
    with tab2:
        st.header("Employee Risk Profile & Retention Plan")
        
        # Select Employee
        emp_ids = [p['Employee_ID'] for p in st.session_state.predictions]
        selected_emp = st.selectbox("Select High-Risk Employee", emp_ids)
        
        if selected_emp:
            # Get employee data
            emp_data = next((p for p in st.session_state.predictions if p["Employee_ID"] == selected_emp), None)
            emp_rec = next((r for r in st.session_state.recommendations if r["Employee_ID"] == selected_emp), None)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(label="Risk Probability", value=f"{emp_data['Risk_Probability']}%", delta=emp_data['Risk_Tier'], delta_color="inverse")
                
                # SHAP Waterfall
                st.subheader("Attrition Risk Drivers (SHAP)")
                
                base_val = emp_data.get("SHAP_Base_Value", 0)
                impacts = emp_data.get("SHAP_All_Impacts", [])
                
                if impacts:
                    import plotly.graph_objects as go
                    # Sort impacts by absolute value
                    sorted_all = sorted(impacts, key=lambda x: abs(x["impact"]), reverse=True)
                    # Keep top 7, group rest as "Other"
                    top_impacts = sorted_all[:7]
                    other_impact = sum([x["impact"] for x in sorted_all[7:]])
                    
                    if abs(other_impact) > 0.01:
                        top_impacts.append({"feature": "Other features", "impact": other_impact})
                        
                    measures = ["absolute"] + ["relative"] * len(top_impacts) + ["total"]
                    x_vals = ["Base Expected"] + [i["feature"] for i in top_impacts] + ["Predicted Risk"]
                    y_vals = [base_val] + [i["impact"] for i in top_impacts] + [0]
                    
                    fig_waterfall = go.Figure(go.Waterfall(
                        orientation="v",
                        measure=measures,
                        x=x_vals,
                        textposition="outside",
                        y=y_vals,
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))
                    fig_waterfall.update_layout(
                        showlegend=False,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                else:
                    for driver in emp_data['Top_Drivers']:
                        st.write(f"- **{driver['feature']}** ({driver['impact_direction']})")
                    
            with col2:
                st.subheader("AI-Generated Retention Action Plan")
                if emp_rec:
                    st.markdown(emp_rec['Action_Plan'])
                    
                    # Download button for Word Doc
                    doc_path = f"reports/ActionPlan_{selected_emp}.docx"
                    if os.path.exists(doc_path):
                        with open(doc_path, "rb") as file:
                            st.download_button(
                                label="📥 Download Word Document",
                                data=file,
                                file_name=f"RetentionPlan_{selected_emp}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                else:
                    st.info("Action plan was not generated for this employee (likely due to API limits in prototype).")
    # --- TAB 3: MODEL METRICS ---
    with tab3:
        st.header("Model Performance Metrics")
        try:
            import json
            with open("model_metrics.json", "r") as f:
                metrics = json.load(f)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{metrics.get('Accuracy', 0):.2%}")
            c2.metric("F1 Score", f"{metrics.get('F1_Score', 0):.3f}")
            c3.metric("AUC-ROC", f"{metrics.get('AUC_ROC', 0):.3f}")
            
            st.subheader("Confusion Matrix")
            cm = metrics.get("Confusion_Matrix")
            if cm:
                fig_cm = px.imshow(cm, text_auto=True, 
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=['Stayed', 'Left'], y=['Stayed', 'Left'],
                                   color_continuous_scale="Blues")
                st.plotly_chart(fig_cm, use_container_width=True)
        except Exception as e:
            st.info("Model metrics are not available. Run the ML pipeline to generate them.")

else:
    st.info("Please upload a CSV file and run the analysis to view the dashboard.")
