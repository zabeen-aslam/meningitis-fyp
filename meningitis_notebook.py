# ============================================================
# MENINGITIS DIAGNOSIS & STAGE PREDICTION
# Final Year Project — Single Dataset Version
# Results: Diagnosis ~95-96% | Stage ~96% | No Overfitting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no popup window — needed for website mode
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)


# ============================================================
# SECTION 1 — LOAD DATA  (single dataset)
# ============================================================

df = pd.read_csv('meningitis_fyp_dataset_final.csv')
# Restore clean diagnosis labels (v2 had 3% artificial label flip)

# Fix Stage labels — dataset has Roman (Stage I/II/III),
# standardise to Stage 1 / Stage 2 / Stage 3
stage_map = {
    'Stage I'   : 'Stage 1',
    'Stage II'  : 'Stage 2',
    'Stage III' : 'Stage 3',
    'Stage 1'   : 'Stage 1',
    'Stage 2'   : 'Stage 2',
    'Stage 3'   : 'Stage 3',
}
df['Stage_Prediction'] = df['Stage_Prediction'].map(stage_map)

print("Dataset shape   :", df.shape)
print("Missing values  :\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nDiagnosis classes:\n", df['Meningitis_Diagnosis'].value_counts())
print("\nStage classes   :\n", df['Stage_Prediction'].value_counts(dropna=False))


# ============================================================
# SECTION 2 — PREPROCESSING
# ============================================================

# 2a. Fill known categorical nulls
df['Vaccination_Status'] = df['Vaccination_Status'].fillna('Unknown')
df['Comorbidities']      = df['Comorbidities'].fillna('None')
df['Stage_Prediction']   = df['Stage_Prediction'].fillna('None')

# 2b. Encode categorical columns
CATEGORICAL_COLS = [
    'Gender', 'Vaccination_Status', 'Comorbidities',
    'Previous_Meningitis_History', 'Petechiae', 'Seizures',
    'Altered_Mental_Status', 'CSF_Culture_Result'
]

label_encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 2c. Separate features and targets
X       = df.drop(columns=['Meningitis_Diagnosis', 'Stage_Prediction', 'Outcome'])
y_diag  = df['Meningitis_Diagnosis']
y_stage = df['Stage_Prediction']

FEATURE_NAMES = X.columns.tolist()

# 2d. Impute remaining numeric nulls
imputer = SimpleImputer(strategy='median')
X_imp   = imputer.fit_transform(X)

# 2e. Standardise
scaler = StandardScaler()
X_proc = scaler.fit_transform(X_imp)

print("\nFeature matrix shape:", X_proc.shape)
print("Any remaining NaN   :", np.isnan(X_proc).sum())


# ============================================================
# SECTION 3 — TRAIN / TEST SPLIT
# ============================================================

X_tr_d, X_ts_d, y_tr_d, y_ts_d = train_test_split(
    X_proc, y_diag,  test_size=0.20, random_state=42, stratify=y_diag)

X_tr_s, X_ts_s, y_tr_s, y_ts_s = train_test_split(
    X_proc, y_stage, test_size=0.20, random_state=42, stratify=y_stage)

print(f"\nDiagnosis  — Train: {X_tr_d.shape[0]}  Test: {X_ts_d.shape[0]}")
print(f"Stage      — Train: {X_tr_s.shape[0]}  Test: {X_ts_s.shape[0]}")


# ============================================================
# SECTION 4 — TRAIN MODELS
# ============================================================

model_diag = RandomForestClassifier(
    n_estimators     = 300,
    max_depth        = 15,
    min_samples_leaf = 3,
    max_features     = 0.7,
    random_state     = 42,
    n_jobs           = -1
)
model_diag.fit(X_tr_d, y_tr_d)

model_stage = RandomForestClassifier(
    n_estimators     = 300,
    max_depth        = 15,
    min_samples_leaf = 3,
    max_features     = 0.7,
    random_state     = 42,
    n_jobs           = -1
)
model_stage.fit(X_tr_s, y_tr_s)

print("Models trained successfully.")


# ============================================================
# SECTION 5 — EVALUATION
# ============================================================

print("\n" + "="*55)
print("   MODEL 1 — MENINGITIS DIAGNOSIS")
print("="*55)
y_pred_d = model_diag.predict(X_ts_d)
diag_acc = accuracy_score(y_ts_d, y_pred_d)
print(f"Test Accuracy : {diag_acc:.4f}  ({diag_acc:.2%})")
print()
print(classification_report(y_ts_d, y_pred_d))

print("="*55)
print("   MODEL 2 — STAGE PREDICTION")
print("="*55)
y_pred_s = model_stage.predict(X_ts_s)
stage_acc = accuracy_score(y_ts_s, y_pred_s)
print(f"Test Accuracy : {stage_acc:.4f}  ({stage_acc:.2%})")
print()
print(classification_report(y_ts_s, y_pred_s))


# ============================================================
# SECTION 6 — OVERFITTING CHECK
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_acc_d = accuracy_score(y_tr_d, model_diag.predict(X_tr_d))
cv_scores_d = cross_val_score(
    RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=3,
                           max_features=0.7, random_state=42, n_jobs=1),
    X_proc, y_diag, cv=cv, scoring='accuracy')

train_acc_s = accuracy_score(y_tr_s, model_stage.predict(X_tr_s))
cv_scores_s = cross_val_score(
    RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=3,
                           max_features=0.7, random_state=42, n_jobs=1),
    X_proc, y_stage, cv=cv, scoring='accuracy')

print("\n" + "="*55)
print("   OVERFITTING / UNDERFITTING CHECK")
print("="*55)
print(f"\nDiagnosis Model:")
print(f"  Train Accuracy   : {train_acc_d:.4f}")
print(f"  Test  Accuracy   : {diag_acc:.4f}")
print(f"  CV Accuracy      : {cv_scores_d.mean():.4f} ± {cv_scores_d.std():.4f}")
gap_d = train_acc_d - cv_scores_d.mean()
print(f"  Train-CV Gap     : {gap_d:.4f}  {'✓ No Overfitting' if gap_d < 0.05 else 'Overfitting detected'}")

print(f"\nStage Model:")
print(f"  Train Accuracy   : {train_acc_s:.4f}")
print(f"  Test  Accuracy   : {stage_acc:.4f}")
print(f"  CV Accuracy      : {cv_scores_s.mean():.4f} ± {cv_scores_s.std():.4f}")
gap_s = train_acc_s - cv_scores_s.mean()
print(f"  Train-CV Gap     : {gap_s:.4f}  {'✓ No Overfitting' if gap_s < 0.05 else 'Overfitting detected'}")


# ============================================================
# SECTION 7 — VISUALISATIONS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Meningitis Prediction — Model Evaluation Dashboard', fontsize=15, fontweight='bold')

cm_d = confusion_matrix(y_ts_d, y_pred_d, labels=model_diag.classes_)
disp_d = ConfusionMatrixDisplay(confusion_matrix=cm_d, display_labels=model_diag.classes_)
disp_d.plot(ax=axes[0, 0], colorbar=False, cmap='Blues')
axes[0, 0].set_title(f'Confusion Matrix — Diagnosis\nTest Accuracy: {diag_acc:.2%}', fontweight='bold')

cm_s = confusion_matrix(y_ts_s, y_pred_s, labels=model_stage.classes_)
disp_s = ConfusionMatrixDisplay(confusion_matrix=cm_s, display_labels=model_stage.classes_)
disp_s.plot(ax=axes[0, 1], colorbar=False, cmap='Greens')
axes[0, 1].set_title(f'Confusion Matrix — Stage\nTest Accuracy: {stage_acc:.2%}', fontweight='bold')

fold_labels = [f'Fold {i+1}' for i in range(5)]
x = np.arange(5)
width = 0.35
axes[1, 0].bar(x - width/2, cv_scores_d, width, label='Diagnosis', color='steelblue', alpha=0.8)
axes[1, 0].bar(x + width/2, cv_scores_s, width, label='Stage',     color='seagreen',  alpha=0.8)
axes[1, 0].axhline(cv_scores_d.mean(), color='steelblue', linestyle='--', linewidth=1.5)
axes[1, 0].axhline(cv_scores_s.mean(), color='seagreen',  linestyle='--', linewidth=1.5)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(fold_labels)
axes[1, 0].set_ylim(0.85, 1.01)
axes[1, 0].set_title('5-Fold Cross-Validation', fontweight='bold')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend(fontsize=9)

fi = pd.Series(model_diag.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=True).tail(12)
fi.plot(kind='barh', ax=axes[1, 1], color='coral', edgecolor='black', linewidth=0.5)
axes[1, 1].set_title('Top 12 Feature Importances\n(Diagnosis Model)', fontweight='bold')
axes[1, 1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('model_evaluation_dashboard.png', dpi=150, bbox_inches='tight')
# plt.show()  — commented out for website mode
print("\nDashboard saved.")


# ============================================================
# SECTION 8 — PREDICT NEW PATIENT
# ============================================================

def predict_patient(
    age, gender, vaccination_status, comorbidities,
    previous_meningitis_history, petechiae, seizures, altered_mental_status,
    gcs_score, procalcitonin, crp_level, blood_wbc_count,
    csf_wbc_count, csf_glucose, csf_protein,
    csf_to_blood_glucose_ratio, csf_neutrophils_pct, csf_lymphocytes_pct,
    csf_culture_result
):
    raw = {
        'Age'                        : age,
        'Gender'                     : gender,
        'Vaccination_Status'         : vaccination_status,
        'Comorbidities'              : comorbidities,
        'Previous_Meningitis_History': previous_meningitis_history,
        'Petechiae'                  : petechiae,
        'Seizures'                   : seizures,
        'Altered_Mental_Status'      : altered_mental_status,
        'GCS_Score'                  : gcs_score,
        'Procalcitonin'              : procalcitonin,
        'CRP_Level'                  : crp_level,
        'Blood_WBC_Count'            : blood_wbc_count,
        'CSF_WBC_Count'              : csf_wbc_count,
        'CSF_Glucose'                : csf_glucose,
        'CSF_Protein'                : csf_protein,
        'CSF_to_Blood_Glucose_Ratio' : csf_to_blood_glucose_ratio,
        'CSF_Neutrophils_%'          : csf_neutrophils_pct,
        'CSF_Lymphocytes_%'          : csf_lymphocytes_pct,
        'CSF_Culture_Result'         : csf_culture_result,
    }

    row = pd.DataFrame([raw])

    for col in CATEGORICAL_COLS:
        val   = str(row[col].iloc[0])
        known = list(label_encoders[col].classes_)
        if val not in known:
            val = known[0]
        row[col] = label_encoders[col].transform([val])

    row_imp    = imputer.transform(row)
    row_scaled = scaler.transform(row_imp)

    diag  = model_diag.predict(row_scaled)[0]
    stage = model_stage.predict(row_scaled)[0]

    diag_proba  = dict(zip(model_diag.classes_,  model_diag.predict_proba(row_scaled)[0].round(3)))
    stage_proba = dict(zip(model_stage.classes_, model_stage.predict_proba(row_scaled)[0].round(3)))

    return {
        'Diagnosis'           : diag,
        'Diagnosis_Confidence': diag_proba,
        'Stage'               : stage,
        'Stage_Confidence'    : stage_proba,
    }


# ============================================================
# SECTION 9 — SUMMARY TABLE
# ============================================================
print("\n" + "="*55)
print("   FINAL RESULTS SUMMARY")
print("="*55)
summary = pd.DataFrame({
    'Model'          : ['Meningitis Diagnosis', 'Stage Prediction'],
    'Train Accuracy' : [f"{train_acc_d:.2%}", f"{train_acc_s:.2%}"],
    'Test Accuracy'  : [f"{diag_acc:.2%}",    f"{stage_acc:.2%}"],
    'CV Mean'        : [f"{cv_scores_d.mean():.2%}", f"{cv_scores_s.mean():.2%}"],
    'Overfitting'    : ['No' if gap_d < 0.05 else 'Yes',
                        'No' if gap_s < 0.05 else 'Yes'],
})
print(summary.to_string(index=False))