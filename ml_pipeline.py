import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
import json

def load_data(file_path='mock_hr_data.csv'):
    """Loads the HR dataset."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Validates schema and handles missing values if any."""
    # In our dummy data there are no NaNs, but in reality we would handle them here.
    df = df.dropna()
    
    # Ensure types are correct
    numeric_cols = ['Age', 'Tenure_Years', 'Salary_USD', 'Performance_Rating', 
                    'Manager_Rating', 'Leave_Days_Taken', 'Avg_Working_Hours_Week', 
                    'Months_Since_Last_Promotion', 'Engagement_Index']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df.dropna()

def feature_engineering(df):
    """Derives predictive features."""
    df = df.copy()
    
    # 1. Tenure Buckets
    bins = [0, 2, 5, 10, 50]
    labels = ['0-2', '3-5', '6-10', '10+']
    df['Tenure_Bucket'] = pd.cut(df['Tenure_Years'], bins=bins, labels=labels, right=False)
    
    # 2. Salary Ratio (Compared to department average)
    dept_avg_salary = df.groupby('Department')['Salary_USD'].transform('mean')
    df['Salary_Ratio_Dept'] = df['Salary_USD'] / dept_avg_salary
    
    # 3. Leave Anomaly (Too high or too low compared to role average)
    role_avg_leave = df.groupby('Role')['Leave_Days_Taken'].transform('mean')
    df['Leave_Anomaly_Score'] = abs(df['Leave_Days_Taken'] - role_avg_leave)
    
    # Drop Employee_ID and Name as they are not predictive
    df = df.drop(['Employee_ID', 'Name'], axis=1)
    
    return df

def preprocess_data(df):
    """Encodes categorical variables and scales numerical features."""
    # Separate target
    y = df['Attrition']
    X = df.drop('Attrition', axis=1)
    
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    # Encode categorical
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
    # Scale numerical
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    return X, y, scaler, label_encoders, cat_cols, num_cols

def train_model(X_train, y_train):
    """Trains an XGBoost classifier."""
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Calculates accuracy, F1, and AUC-ROC."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred),
        'AUC_ROC': roc_auc_score(y_test, y_prob),
        'Confusion_Matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics

def run_pipeline():
    print("Starting ML Pipeline...")
    
    # 1. Load Data
    df = load_data('mock_hr_data.csv')
    print(f"Data loaded: {df.shape}")
    
    # 2. Clean & Feature Engineer
    df_clean = clean_data(df)
    df_features = feature_engineering(df_clean)
    print(f"Features engineered. Shape: {df_features.shape}")
    
    # 3. Preprocess
    X, y, scaler, label_encoders, cat_cols, num_cols = preprocess_data(df_features)
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Train Model
    model = train_model(X_train, y_train)
    print("Model training complete.")
    
    # 6. Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    # 7. Save Artifacts
    joblib.dump(model, 'attrition_xgboost_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    # Save encoders and columns info for inference
    preprocessing_info = {
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'label_encoders_classes': {col: le.classes_.tolist() for col, le in label_encoders.items()}
    }
    with open('preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f)
        
    print("Artifacts saved: attrition_xgboost_model.pkl, scaler.pkl, preprocessing_info.json")
    
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f)
    print("Metrics saved to model_metrics.json")

if __name__ == "__main__":
    run_pipeline()
