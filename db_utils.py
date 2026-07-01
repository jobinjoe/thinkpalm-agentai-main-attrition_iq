import sqlite3
import pandas as pd
from datetime import datetime
import json

DB_PATH = "attrition_history.db"

def init_db():
    """Initializes the SQLite database and creates the necessary tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create a table to store employee predictions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp TEXT,
            employee_id TEXT,
            risk_probability REAL,
            risk_tier TEXT,
            top_drivers TEXT,
            action_plan TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_predictions(predictions, recommendations):
    """
    Saves a batch of predictions and recommendations to the SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Use a single timestamp for the entire batch run
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Convert recommendations list into a dictionary for quick lookup
    rec_dict = {r["Employee_ID"]: r.get("Action_Plan", "") for r in recommendations}
    
    for pred in predictions:
        emp_id = str(pred["Employee_ID"])
        risk_prob = pred.get("Risk_Probability", 0.0)
        risk_tier = pred.get("Risk_Tier", "Unknown")
        
        # Serialize top drivers back to a JSON string or simple text for storage
        drivers = json.dumps([d["feature"] for d in pred.get("Top_Drivers", [])])
        
        # Fetch action plan if it exists
        action_plan = rec_dict.get(emp_id, "No action plan generated.")
        
        cursor.execute('''
            INSERT INTO historical_predictions 
            (run_timestamp, employee_id, risk_probability, risk_tier, top_drivers, action_plan)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (run_timestamp, emp_id, risk_prob, risk_tier, drivers, action_plan))
        
    conn.commit()
    conn.close()

def get_historical_predictions():
    """Fetches all historical predictions from the database as a pandas DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    
    # Check if table has data
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_predictions'")
    if not cursor.fetchone():
        conn.close()
        return pd.DataFrame()
        
    df = pd.read_sql_query("SELECT * FROM historical_predictions ORDER BY run_timestamp DESC", conn)
    conn.close()
    return df
