import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("nn_readmission_model.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="🔮 Patient Readmission Predictor", layout="centered")
st.title("🏥 Predict Patient Readmission")
st.markdown("This app predicts if a diabetic patient is likely to be readmitted within 30 days.")

# Encodings (from preprocessing)
age_mapping = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3,
    "[40-50)": 4, "[50-60)": 5, "[60-70)": 6, "[70-80)": 7,
    "[80-90)": 8, "[90-100)": 9
}
gender_mapping = {"Male": 1, "Female": 0}
race_mapping = {"Caucasian": 0, "AfricanAmerican": 1, "Other": 2, "Asian": 3, "Hispanic": 4}
change_mapping = {"No": 0, "Ch": 1}
diabetes_med_mapping = {"No": 0, "Yes": 1}
meds_mapping = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}

admission_type_labels = {
    0: "Emergency", 1: "Urgent", 2: "Elective", 3: "Newborn",
    4: "Not Available", 5: "Trauma Center", 6: "Not Mapped", 7: "Unknown"
}
diagnosis_labels = {
    "Diabetes (250.xx)": 42,
    "Hypertension (401.xx)": 18,
    "Ischemic Heart Disease (414.xx)": 27,
    "Asthma (493.xx)": 36,
    "Pneumonia (486.xx)": 22,
    "Chronic Kidney Disease (585.xx)": 57,
    "Obesity (278.xx)": 13,
    "Other": 0
}
# UI Inputs
age = age_mapping[st.selectbox("Age", list(age_mapping.keys()))]
gender = gender_mapping[st.selectbox("Gender", list(gender_mapping.keys()))]
race = race_mapping[st.selectbox("Race", list(race_mapping.keys()))]
admission_type = st.selectbox("Admission Type", options=list(admission_type_labels.keys()), format_func=lambda x: admission_type_labels[x])
discharge_id = st.slider("Discharge Disposition ID", 0, 25, 1)
admission_src = st.slider("Admission Source ID", 0, 20, 1)


# Diagnoses (via dropdown)
diag_1 = diagnosis_labels[st.selectbox("Diagnosis 1", list(diagnosis_labels.keys()), index=0)]
diag_2 = diagnosis_labels[st.selectbox("Diagnosis 2", list(diagnosis_labels.keys()), index=1)]
diag_3 = diagnosis_labels[st.selectbox("Diagnosis 3", list(diagnosis_labels.keys()), index=2)]
# Hospital stay & procedures
time_in_hospital = st.slider("Time in Hospital (days)", 1, 30, 5)
num_lab_procedures = st.slider("Number of Lab Procedures", 0, 150, 40)
num_procedures = st.slider("Number of Procedures", 0, 10, 1)
num_medications = st.slider("Number of Medications", 0, 100, 10)
number_outpatient = st.slider("Number of Outpatient visits", 0, 20, 0)
number_emergency = st.slider("Number of Emergency visits", 0, 20, 0)
number_inpatient = st.slider("Number of Inpatient visits", 0, 20, 0)
number_diagnoses = st.slider("Number of Diagnoses", 0, 16, 5)

# Medications
metformin = meds_mapping[st.selectbox("Metformin", list(meds_mapping.keys()))]
insulin = meds_mapping[st.selectbox("Insulin", list(meds_mapping.keys()))]
glipizide = meds_mapping[st.selectbox("Glipizide", list(meds_mapping.keys()))]
glyburide = meds_mapping[st.selectbox("Glyburide", list(meds_mapping.keys()))]
glimepiride = meds_mapping[st.selectbox("Glimepiride", list(meds_mapping.keys()))]
repaglinide = meds_mapping[st.selectbox("Repaglinide", list(meds_mapping.keys()))]
nateglinide = meds_mapping[st.selectbox("Nateglinide", list(meds_mapping.keys()))]
chlorpropamide = meds_mapping[st.selectbox("Chlorpropamide", list(meds_mapping.keys()))]
acetohexamide = meds_mapping[st.selectbox("Acetohexamide", list(meds_mapping.keys()))]
tolbutamide = meds_mapping[st.selectbox("Tolbutamide", list(meds_mapping.keys()))]
pioglitazone = meds_mapping[st.selectbox("Pioglitazone", list(meds_mapping.keys()))]
rosiglitazone = meds_mapping[st.selectbox("Rosiglitazone", list(meds_mapping.keys()))]
acarbose = meds_mapping[st.selectbox("Acarbose", list(meds_mapping.keys()))]
miglitol = meds_mapping[st.selectbox("Miglitol", list(meds_mapping.keys()))]
troglitazone = meds_mapping[st.selectbox("Troglitazone", list(meds_mapping.keys()))]
tolazamide = meds_mapping[st.selectbox("Tolazamide", list(meds_mapping.keys()))]
glyburide_metformin = meds_mapping[st.selectbox("Glyburide-Metformin", list(meds_mapping.keys()))]
glipizide_metformin = meds_mapping[st.selectbox("Glipizide-Metformin", list(meds_mapping.keys()))]
glimepiride_pioglitazone = meds_mapping[st.selectbox("Glimepiride-Pioglitazone", list(meds_mapping.keys()))]
metformin_rosiglitazone = meds_mapping[st.selectbox("Metformin-Rosiglitazone", list(meds_mapping.keys()))]
metformin_pioglitazone = meds_mapping[st.selectbox("Metformin-Pioglitazone", list(meds_mapping.keys()))]

# Treatment change & medication
change = change_mapping[st.selectbox("Change in medications?", list(change_mapping.keys()))]
diabetes_med = diabetes_med_mapping[st.selectbox("Is on diabetes medication?", list(diabetes_med_mapping.keys()))]

# Create input vector (match training feature order)
input_data = np.array([[

    admission_type, discharge_id, admission_src,
    race, gender, age,
    diag_1, diag_2, diag_3,
    metformin, repaglinide, nateglinide, chlorpropamide,
    glimepiride, acetohexamide, glipizide, glyburide,
    tolbutamide, pioglitazone, rosiglitazone, acarbose,
    miglitol, troglitazone, tolazamide, insulin,
    glyburide_metformin, glipizide_metformin, glimepiride_pioglitazone,
    metformin_rosiglitazone, metformin_pioglitazone,
    change, diabetes_med,
    time_in_hospital, num_lab_procedures, num_procedures,
    num_medications, number_outpatient, number_emergency,
    number_inpatient, number_diagnoses

]])
# Predict
if st.button("🔮 Predict Readmission"):
    input_scaled = scaler.transform(input_data)
    prob = model.predict(input_scaled)[0][1]
    st.subheader("🧠 Prediction Result")
    st.write(f"**Probability of Readmission:** `{prob:.2f}`")

    if prob >= 0.5:
        st.error("🔴 Patient is likely to be readmitted within 30 days.")
    else:
        st.success("🟢 Patient is unlikely to be readmitted within 30 days.")
