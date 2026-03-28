from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# ── Models Load Karo ──────────────────────────────────────
with open('model_diag.pkl',     'rb') as f: model_diag     = pickle.load(f)
with open('model_stage.pkl',    'rb') as f: model_stage    = pickle.load(f)
with open('imputer.pkl',        'rb') as f: imputer        = pickle.load(f)
with open('scaler.pkl',         'rb') as f: scaler         = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f: label_encoders = pickle.load(f)
with open('feature_names.pkl',  'rb') as f: FEATURE_NAMES  = pickle.load(f)

CATEGORICAL_COLS = [
    'Gender', 'Vaccination_Status', 'Comorbidities',
    'Previous_Meningitis_History', 'Petechiae', 'Seizures',
    'Altered_Mental_Status', 'CSF_Culture_Result'
]

# ── Health Check Route ────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'API is running ✓'})

# ── Prediction Route ──────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        row  = pd.DataFrame([data])

        # Categorical encoding
        for col in CATEGORICAL_COLS:
            val   = str(row[col].iloc[0])
            known = list(label_encoders[col].classes_)
            if val not in known:
                val = known[0]
            row[col] = label_encoders[col].transform([val])

        # Impute + Scale
        row_imp    = imputer.transform(row)
        row_scaled = scaler.transform(row_imp)

        # Predict
        diag       = model_diag.predict(row_scaled)[0]
        stage      = model_stage.predict(row_scaled)[0]
        diag_proba = dict(zip(model_diag.classes_,
                              model_diag.predict_proba(row_scaled)[0].round(3)))
        stage_proba= dict(zip(model_stage.classes_,
                              model_stage.predict_proba(row_scaled)[0].round(3)))

        return jsonify({
            'success'             : True,
            'diagnosis'           : diag,
            'diagnosis_confidence': diag_proba,
            'stage'               : stage,
            'stage_confidence'    : stage_proba
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)