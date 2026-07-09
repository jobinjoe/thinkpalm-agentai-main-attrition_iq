# AttritionIQ: Architecture & Write-Up

## System Overview
AttritionIQ is an advanced HR analytics platform that bridges traditional Machine Learning with modern Generative AI to combat employee turnover. Rather than simply flagging employees as "high risk," AttritionIQ employs a multi-agent workflow to diagnose *why* an employee might leave and immediately proposes an actionable retention strategy.

## Architecture & Workflow

The core of AttritionIQ is powered by **LangGraph**, which orchestrates a sequence of specialized AI agents. The workflow is triggered via a **Streamlit** user interface when an HR professional uploads a dataset. 

The agentic pipeline consists of the following sequential nodes:

1. **Data Ingestion Agent**: 
   Validates the uploaded CSV schema, cleans the data, and ensures it is ready for downstream processing.
2. **Feature Engineering Agent**: 
   Derives complex predictive features from raw data, such as tenure buckets, salary ratios compared to department averages, and leave anomalies.
3. **Prediction & Scoring Agent**: 
   Applies a pre-trained **XGBoost** classifier (trained via `ml_pipeline.py`) to the engineered dataset. It outputs a probability score (flight risk) for each employee and persists the results to a local **SQLite** database.
4. **Recommendation Agent**: 
   Acts as an AI HR Consultant. For every employee flagged as high-risk, this agent queries a Generative LLM (**Llama 3** via **Groq**) with the employee's specific risk factors (e.g., low salary ratio, high tenure without promotion) to generate a customized retention action plan.
5. **Report Generation Agent**: 
   Synthesizes the predictions and LLM recommendations into a cohesive, styled HTML report.
6. **Alert Agent**: 
   Handles distribution, capable of sending the final HTML report to the HR team via SMTP email.

## Technology Stack
- **Frontend / UI**: Streamlit, Plotly
- **Workflow Orchestration**: LangGraph
- **Generative AI**: Groq API (Llama-3-8b-instant), LangChain
- **Machine Learning**: XGBoost, Scikit-Learn, Pandas
- **Storage**: SQLite

## Business Impact
By automating the data pipeline and integrating LLM-based reasoning, AttritionIQ reduces the time HR teams spend manually analyzing spreadsheets. It shifts the paradigm from *reactive* reporting to *proactive* intervention, ensuring that managers have concrete, personalized action plans ready before an employee decides to resign.
