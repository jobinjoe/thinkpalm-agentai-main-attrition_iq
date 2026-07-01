# AttritionIQ

Agentic Attrition Prediction Platform built using LangGraph, Groq (Llama 3), Streamlit, XGBoost, and SHAP.

## Features
- **Data Ingestion & Validation Agent**: Cleans and validates HR CSV files.
- **Feature Engineering Agent**: Derives predictive features and logs transformations.
- **Prediction & Scoring Agent**: XGBoost model predicting attrition risk for each employee.
- **Explainability Agent**: SHAP integration extracting the top driving factors for predictions.
- **Retention Recommendation Agent**: LLM (Groq) generates a personalized 3-point action plan.
- **Report & Alert Agent**: Generates HTML reports and sends notifications via Slack and SMTP.

## Setup Instructions

1. **Clone the repository** (if applicable) or download the files.
2. **Create a virtual environment** and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables**:
   Create a `.env` file in the root directory and add the following keys:
   ```env
   GROQ_API_KEY=your_groq_api_key
   
   # Optional Alerting Variables
   SLACK_WEBHOOK_URL=your_slack_webhook
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your_email@gmail.com
   SMTP_PASS=your_app_password
   ```
5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
6. **(Optional) Re-train the model**:
   ```bash
   python ml_pipeline.py
   ```

## Deliverables Included
- `mock_hr_data.csv`: Sample data
- `feature_importance.png`: Generated chart of model features
- `reports/Attrition_Risk_Report.html`: Generated Weekly Report
- `attrition_history.db`: SQLite database logging history
