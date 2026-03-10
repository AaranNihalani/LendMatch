import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.prediction_service import PredictionService

def test_prediction():
    print("Testing Prediction Service...")
    service = PredictionService(models_dir="models")
    
    input_data = {
        "loan_amount": 15000.0,
        "annual_inc": 60000.0,
        "fico_score": 700.0,
        "dti": 15.0,
        "state": "CA",
        "term": 36,
        "emp_length": "10+ years",
        "purpose": "debt_consolidation",
        "home_ownership": "MORTGAGE",
        "revol_bal": 10000.0,
        "total_acc": 22.0
    }
    
    print("Input:", input_data)
    result = service.predict(input_data)
    print("Result:", result)
    
    assert "approval_probability" in result
    assert "is_approved" in result
    assert "predicted_interest_rate" in result
    assert "default_probability" in result
    assert "approval_probability" in result
    assert "offers" in result
    print("Test Passed!")


if __name__ == "__main__":
    test_prediction()
