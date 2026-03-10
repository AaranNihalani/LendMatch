import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.prediction_service import PredictionService

app = FastAPI(title="Loan Matching Platform API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Service
# Check if models exist
if os.path.exists("models/approval_model.pkl"):
    try:
        service = PredictionService()
    except Exception as e:
        print(f"WARNING: Failed to initialize PredictionService: {e}")
        service = None
else:
    print("WARNING: Models not found. API running in limited mode.")
    service = None

class LoanApplication(BaseModel):
    loan_amount: float
    annual_inc: float
    fico_score: float
    dti: float
    state: str
    term: int = 36
    emp_length: str = "1 year"
    purpose: str = "debt_consolidation"
    home_ownership: str = "RENT"
    revol_bal: float = 0.0
    total_acc: float = 10.0

@app.get("/health")
def health_check():
    payload = {"status": "ok", "models_loaded": service is not None}
    if service is not None:
        payload.update(
            {
                "has_approval_model": getattr(service, "approval_model", None) is not None,
                "has_default_model": getattr(service, "default_model", None) is not None,
                "has_interest_model": getattr(service, "interest_model", None) is not None,
                "has_approval_preprocessor": getattr(service, "approval_preprocessor", None) is not None,
                "has_full_preprocessor": getattr(service, "full_preprocessor", None) is not None,
                "approval_preprocessor_type": type(getattr(service, "approval_preprocessor", None)).__name__,
                "full_preprocessor_type": type(getattr(service, "full_preprocessor", None)).__name__,
            }
        )
    return payload

@app.post("/predict")
def predict(application: LoanApplication):
    if not service:
        raise HTTPException(status_code=503, detail="Prediction service not available (models not loaded)")
    
    data = application.dict()
    try:
        results = service.predict(data)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Serve Static Files (Frontend)
if not os.path.exists("docs"):
    os.makedirs("docs")

app.mount("/", StaticFiles(directory="docs", html=True), name="static")
