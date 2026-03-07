import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.prediction_service import PredictionService

def test_prediction():
    print("Testing Prediction Service...")
    service = PredictionService(
        models_dir="models",
        artifacts_dir="models/artifacts"
    )
    
    input_data = {
        "annual_inc": 60000.0,
        "loan_amnt": 15000.0,
        "credit_score": 700.0,
        "emp_length": "10+ years",
        "purpose": "debt_consolidation",
        "home_ownership": "MORTGAGE",
        "term": " 36 months",
        "dti": 15.0,
        "revol_bal": 10000.0,
        "revol_util": 50.0
    }
    
    print("Input:", input_data)
    result = service.predict(input_data)
    print("Result:", result)
    
    assert "default_risk" in result
    assert "predicted_interest_rate" in result
    assert "approval_probability" in result
    assert "recommended_lenders" in result
    assert len(result["recommended_lenders"]) > 0
    
    print("Test Passed!")

if __name__ == "__main__":
    test_prediction()
