# Meningitis Prediction System

AI-powered Meningitis Diagnosis and Stage Prediction using Random Forest.

| Model | Accuracy |
|-------|----------|
| Diagnosis | 96.5% |
| Stage | 95.5% |

## How to Run

Install dependencies:
pip install -r requirements.txt

Run API:
uvicorn main:app --reload

Run Website:
streamlit run app.py

## API Endpoint

POST /predict - Predict diagnosis and stage
GET /health - Check if API is running
GET /docs - Interactive API documentation