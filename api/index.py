import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.lendmatch_model import ARTIFACT_PATH, MODEL_CARD_PATH
from src.prediction_service import PredictionService


app = FastAPI(title="LendMatch API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    service = PredictionService()
    load_error = None
except Exception as exc:
    service = None
    load_error = str(exc)


class LoanApplication(BaseModel):
    loan_amount: float = Field(ge=500, le=100000)
    annual_inc: float = Field(ge=1000, le=5000000)
    fico_score: float = Field(ge=300, le=850)
    dti: float = Field(ge=0, le=80)
    state: str = Field(min_length=2, max_length=2)
    term: int = Field(default=36, ge=12, le=84)
    emp_length: str = "2 years"
    purpose: str = "debt_consolidation"
    home_ownership: str = "RENT"
    verification_status: str = "Source Verified"
    application_type: str = "Individual"
    revol_bal: Optional[float] = Field(default=None, ge=0)
    revol_util: Optional[float] = Field(default=None, ge=0)
    total_acc: Optional[float] = Field(default=None, ge=0)
    open_acc: Optional[float] = Field(default=None, ge=0)
    delinq_2yrs: Optional[float] = Field(default=None, ge=0)
    inq_last_6mths: Optional[float] = Field(default=None, ge=0)
    pub_rec: Optional[float] = Field(default=None, ge=0)
    credit_history_years: Optional[float] = Field(default=None, ge=0)


@app.get("/health")
def health_check():
    model_card = None
    if MODEL_CARD_PATH.exists():
        model_card = json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
    return {
        "status": "ok" if service else "model_unavailable",
        "models_loaded": service is not None,
        "artifact": str(ARTIFACT_PATH),
        "load_error": load_error,
        "metrics": (model_card or {}).get("metrics"),
    }


@app.get("/model-card")
def model_card():
    if not MODEL_CARD_PATH.exists():
        raise HTTPException(status_code=404, detail="Model card not found. Train the models first.")
    return json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(application: LoanApplication):
    if service is None:
        raise HTTPException(status_code=503, detail=load_error or "Prediction service is not available.")
    try:
        if hasattr(application, "model_dump"):
            payload = application.model_dump()
        else:
            payload = application.dict()
        return service.predict(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


docs_dir = Path(__file__).resolve().parents[1] / "docs"
app.mount("/", StaticFiles(directory=str(docs_dir), html=True), name="static")
