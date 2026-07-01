import shap
import joblib
import pandas as pd
import numpy as np
import json

class SHAPExplainer:
    def __init__(self, model_path='attrition_xgboost_model.pkl'):
        self.model = joblib.load(model_path)
        # TreeExplainer is fast for XGBoost
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain_prediction(self, feature_names, X_instance):
        """
        Takes a single employee's preprocessed features and returns the top driving factors.
        """
        # Calculate SHAP values for the instance
        shap_values = self.explainer.shap_values(X_instance)
        
        # If binary classification, shap_values might be a list or a single array
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0] # Get values for positive class (attrition = 1)
        else:
            # XGBoost TreeExplainer usually returns just one array for binary logloss
            shap_vals = shap_values[0]
            
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
            
        # Create a dictionary of feature names to their impact scores
        feature_impacts = {feature_names[i]: float(shap_vals[i]) for i in range(len(feature_names))}
        
        # Sort by absolute impact to find top drivers
        sorted_impacts = sorted(feature_impacts.items(), key=lambda item: abs(item[1]), reverse=True)
        
        # Format the top 3 drivers for the LLM
        top_drivers = []
        for feature, impact in sorted_impacts[:3]:
            direction = "increased" if impact > 0 else "decreased"
            # Normalize impact to a rough percentage-like score for readability (optional, doing raw for now)
            top_drivers.append({
                "feature": feature,
                "impact_direction": direction,
                "raw_impact_score": round(impact, 3)
            })
            
        # Also return all feature impacts for the waterfall chart
        feature_impacts_list = [{"feature": f, "impact": float(v)} for f, v in feature_impacts.items()]
        
        return {
            "base_value": float(base_value),
            "top_drivers": top_drivers,
            "all_impacts": feature_impacts_list
        }

if __name__ == "__main__":
    # Quick test to ensure it works
    with open('preprocessing_info.json', 'r') as f:
        info = json.load(f)
    feature_names = info['cat_cols'] + info['num_cols']
    
    # Dummy instance based on the number of features
    dummy_x = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
    
    explainer = SHAPExplainer()
    res = explainer.explain_prediction(feature_names, dummy_x)
    print(json.dumps(res, indent=2))
