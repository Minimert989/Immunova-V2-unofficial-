# api.py

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
import torch
from treatment_response.fusion_model import FusionPredictor
from survival_analysis.survival_transformer import SurvivalTransformer
from drug_optimization.gnn_model import DrugGNN
from drug_optimization.rl_optimizer import DrugRLAgent
import json

app = FastAPI(title="Immunova API")

# Load models (mocked for demonstration)
fusion_model = FusionPredictor()
survival_model = SurvivalTransformer()
drug_model = DrugGNN(in_channels=32, hidden_channels=64, out_channels=2)

class GenomicsInput(BaseModel):
    data: list

class ImagingInput(BaseModel):
    path: str

class SurvivalInput(BaseModel):
    patient_features: list

class DrugInput(BaseModel):
    compound_features: list

@app.post("/predict/treatment")
def predict_treatment(genomics: GenomicsInput, imaging: ImagingInput):
    result = fusion_model.predict(genomics.data, imaging.path)
    return {"treatment_prediction": result}

@app.post("/predict/survival")
def predict_survival(input: SurvivalInput):
    result = survival_model.predict(input.patient_features)
    return {"survival_prediction": result}

@app.post("/predict/drug")
def predict_drug(input: DrugInput):
    # Placeholder for graph input
    prediction = drug_model(torch.tensor(input.compound_features))
    return {"drug_response": prediction.argmax(dim=1).tolist()}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Immunova API"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
