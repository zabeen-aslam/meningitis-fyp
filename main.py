from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, numpy as np, pandas as pd

with open("model_diag.pkl",     "rb") as f: model_diag     = pickle.load(f)
with open("model_stage.pkl",    "rb") as f: model_stage    = pickle.load(f)
with open("imputer.pkl",        "rb") as f: imputer        = pickle.load(f)
with open("scaler.pkl",         "rb") as f: scaler         = pickle.load(f)
with open("label_encoders.pkl", "rb") as f: label_encoders = pickle.load(f)

CATEGORICAL_COLS = ['Gender','Vaccination_Status','Comorbidities',
                    'Previous_Meningitis_History','Petechiae','Seizures',
                    'Altered_Mental_Status','CSF_Culture_Result']

app = FastAPI(
    title="Meningitis Prediction API",
    description="Random Forest | Diagnosis 96.5% | Stage 95.5%",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class PatientInput(BaseModel):
    age:                         float
    gender:                      str
    vaccination_status:          str
    comorbidities:               str
    previous_meningitis_history: str
    petechiae:                   str
    seizures:                    str
    altered_mental_status:       str
    gcs_score:                   float
    procalcitonin:               float
    crp_level:                   float
    blood_wbc_count:             float
    csf_wbc_count:               float
    csf_glucose:                 float
    csf_protein:                 float
    csf_to_blood_glucose_ratio:  float
    csf_neutrophils_pct:         float
    csf_lymphocytes_pct:         float
    csf_culture_result:          str

def preprocess(p):
    row = pd.DataFrame([{
        'Age': p.age, 'Gender': p.gender,
        'Vaccination_Status': p.vaccination_status,
        'Comorbidities': p.comorbidities,
        'Previous_Meningitis_History': p.previous_meningitis_history,
        'Petechiae': p.petechiae, 'Seizures': p.seizures,
        'Altered_Mental_Status': p.altered_mental_status,
        'GCS_Score': p.gcs_score, 'Procalcitonin': p.procalcitonin,
        'CRP_Level': p.crp_level, 'Blood_WBC_Count': p.blood_wbc_count,
        'CSF_WBC_Count': p.csf_wbc_count, 'CSF_Glucose': p.csf_glucose,
        'CSF_Protein': p.csf_protein,
        'CSF_to_Blood_Glucose_Ratio': p.csf_to_blood_glucose_ratio,
        'CSF_Neutrophils_%': p.csf_neutrophils_pct,
        'CSF_Lymphocytes_%': p.csf_lymphocytes_pct,
        'CSF_Culture_Result': p.csf_culture_result,
    }])
    for col in CATEGORICAL_COLS:
        val = str(row[col].iloc[0])
        known = list(label_encoders[col].classes_)
        if val not in known: val = known[0]
        row[col] = label_encoders[col].transform([val])
    return scaler.transform(imputer.transform(row))

@app.get("/")
def root():
    return {"message": "Meningitis Prediction API is running ✓",
            "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(patient: PatientInput):
    X = preprocess(patient)
    diag        = model_diag.predict(X)[0]
    stage       = model_stage.predict(X)[0]
    diag_proba  = dict(zip(model_diag.classes_,
                           np.round(model_diag.predict_proba(X)[0], 4).tolist()))
    stage_proba = dict(zip(model_stage.classes_,
                           np.round(model_stage.predict_proba(X)[0], 4).tolist()))
    risk = ("Critical" if diag=="Bacterial" and stage=="Stage III" else
            "High"     if diag=="Bacterial"   else
            "Moderate" if diag=="Tuberculous" else
            "Low"      if diag=="Viral"       else "Normal")
    return {
        "diagnosis": diag,
        "diagnosis_confidence": diag_proba,
        "stage": stage,
        "stage_confidence": stage_proba,
        "risk_level": risk,
        "model_info": {
            "algorithm": "Random Forest (300 trees)",
            "diagnosis_accuracy": "96.5%",
            "stage_accuracy": "95.5%"
        }
    }