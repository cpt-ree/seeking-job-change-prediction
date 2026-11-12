import pickle
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / 'model.bin'

class Candidate(BaseModel):
    city: str
    city_development_index: float
    gender: str | None = None
    relevent_experience: bool
    enrolled_university: str | None = None
    education_level: str | None = None
    major_discipline: str | None = None
    experience: int
    company_size: int
    company_type: str | None = None
    last_new_job: int
    training_hours: int

app = FastAPI()

def predict_single(customer):
    customer = {
        key: np.nan if value is None else value
        for key, value in customer.items()
    }

    X = dv.transform([customer])
    dX = xgb.DMatrix(X, feature_names=dv.get_feature_names_out().tolist())

    y_pred = model.predict(dX)

    return float(y_pred[0])

@app.post('/predict')
def predict_endpoint(candidate: Candidate):
    prediction = predict_single(candidate.model_dump())

    return {
        'probability': prediction,
        'would_change_jobs': prediction >= 0.5
    }

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)