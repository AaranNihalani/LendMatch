from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.prediction_service import PredictionService
import os

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
    service = PredictionService()
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
    return {"status": "ok", "models_loaded": service is not None}

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
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
