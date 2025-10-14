# multidisease_streamlit_app.py
"""
Multi-disease Streamlit app (final corrected)
- Landing page with two buttons: Register patient data | Detect disease
- Detect disease -> 4 tabs (PCOS, Thyroid, Anemia, Osteoporosis)
- Models: looks for files in ./models like 'PCOS_RF.pkl'. If not found, returns a dummy prediction.
- Patient registry saved to ./data/patients.csv
"""

import streamlit as st
import pandas as pd
import os
import joblib
import pickle
from datetime import datetime

# Paths
MODELS_DIR = "models"
DATA_DIR = "data"
PATIENTS_FILE = os.path.join(DATA_DIR, "patients.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="SheHealth", layout="wide")


# -----------------------
# Utility functions
# -----------------------
def load_model(path):
    """Try to load a model (joblib or pickle). Return None if not found or failed."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None


def predict_with_model(model, X):
    """If model exists, use it. Otherwise return dummy."""
    if model is None:
        # Dummy: return probability 0.2 and label 'Negative' for testing
        return {"label": "Negative", "probability": 0.2}
    try:
        # If model expects 2D array-like
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([X])[0]
            if len(probs) >= 2:
                prob_pos = probs[1]
            else:
                prob_pos = max(probs)
            label = "Positive" if prob_pos >= 0.5 else "Negative"
            return {"label": label, "probability": float(prob_pos)}
        else:
            pred = model.predict([X])[0]
            # Interpret common outputs
            if isinstance(pred, (int, float)):
                label = "Positive" if pred == 1 else "Negative"
            else:
                label = "Positive" if str(pred).lower() in ("positive", "pos", "1", "true", "yes") else "Negative"
            return {"label": label, "probability": None}
    except Exception as e:
        return {"label": "Error", "probability": None, "error": str(e)}


def append_patient(row: dict):
    """Append a patient row (dict) to patients csv."""
    df = pd.DataFrame([row])
    if os.path.exists(PATIENTS_FILE):
        df.to_csv(PATIENTS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(PATIENTS_FILE, index=False)


def read_patients():
    if os.path.exists(PATIENTS_FILE):
        return pd.read_csv(PATIENTS_FILE)
    return pd.DataFrame()


# -----------------------
# UI: Landing page
# -----------------------
def landing_page():
    st.title("🔬 SheHealth-Multiple Disease Predictor")
    st.markdown(
        """
        Welcome! This site provides:
        - Register patient data
        - Detect disease (PCOS, Thyroid, Anemia, Osteoporosis)
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Register patient data", key="register_btn", use_container_width=True):
            st.session_state.page = "register"

    with col2:
        if st.button("🩺 Detect disease", key="detect_btn", use_container_width=True):
            st.session_state.page = "detect"

    st.markdown("---")
    st.subheader("Available saved patients")
    patients = read_patients()
    if patients.empty:
        st.info("No patients registered yet.")
    else:
        st.dataframe(patients.tail(20))


# -----------------------
# UI: Register patient
# -----------------------
def register_page():
    st.header("Register Patient Data")
    st.write("Fill the form below to register a patient (saved locally).")

    with st.form("patient_form", clear_on_submit=True):
        name = st.text_input("Full name")
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        phone = st.text_input("Phone (optional)")
        notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Save patient")

    if submitted:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "name": name,
            "age": int(age) if age.is_integer() else float(age),
            "gender": gender,
            "phone": phone,
            "notes": notes,
        }
        append_patient(row)
        st.success("Patient saved ✅")
        st.experimental_rerun()

  

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"


# -----------------------
# UI: Detect disease
# -----------------------
def detect_page():
    st.header("Detect disease")
    st.write("Choose a disease tab, enter inputs, and press Predict.")

    tabs = st.tabs(["PCOS", "Thyroid", "Anemia", "Osteoporosis"])

    # Disease definitions with fields (order matters for models)
    # All numeric st.number_input parameters are forced to float types to avoid StreamlitMixedNumericTypesError.
    disease_defs = [
        {
            "name": "PCOS",
            "model_file": os.path.join(MODELS_DIR, "PCOS_RF.pkl"),
            "fields": [
                {"key": "age", "label": "Age", "type": "number", "min": 10.0, "max": 100.0, "value": 25.0, "step": 1.0},
                {"key": "bmi", "label": "BMI", "type": "number", "min": 10.0, "max": 60.0, "value": 24.0, "step": 0.1},
                {"key": "menstrual_irregularity", "label": "Menstrual Irregularity (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "testosterone", "label": "Testosterone Level (ng/dL)", "type": "number", "min": 0.0, "max": 300.0, "value": 40.0, "step": 0.1},
                {"key": "antral_follicle_count", "label": "Antral Follicle Count", "type": "number", "min": 0.0, "max": 50.0, "value": 10.0, "step": 1.0},
            ],
        },
        {
            "name": "Thyroid",
            "model_file": os.path.join(MODELS_DIR, "Thyroid_RF.pkl"),
            "fields": [
                {"key": "age", "label": "Age", "type": "number", "min": 0.0, "max": 120.0, "value": 40.0, "step": 1.0},
                {"key": "sex", "label": "Sex", "type": "select", "options": ["M", "F"], "value": "F"},
                # many boolean flags
                {"key": "on_thyroxine", "label": "On thyroxine (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "query_on_thyroxine", "label": "Query on thyroxine (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "on_antithyroid_medication", "label": "On antithyroid medication (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "sick", "label": "Sick (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "pregnant", "label": "Pregnant (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "thyroid_surgery", "label": "Thyroid surgery (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "i131_treatment", "label": "I131 treatment (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "query_hypothyroid", "label": "Query hypothyroid (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "query_hyperthyroid", "label": "Query hyperthyroid (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "lithium", "label": "Lithium (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "goitre", "label": "Goitre (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "tumor", "label": "Tumor (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "hypopituitary", "label": "Hypopituitary (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "psych", "label": "Psych (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "tsh_measured", "label": "TSH measured (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "Yes"},
                {"key": "tsh", "label": "TSH value", "type": "number", "min": 0.0, "max": 100.0, "value": 2.5, "step": 0.01},
                {"key": "t3_measured", "label": "T3 measured (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "tt4_measured", "label": "TT4 measured (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "tt4", "label": "TT4 value", "type": "number", "min": 0.0, "max": 500.0, "value": 100.0, "step": 0.1},
                {"key": "t4u_measured", "label": "T4U measured (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "t4u", "label": "T4U value", "type": "number", "min": 0.0, "max": 10.0, "value": 1.0, "step": 0.01},
                {"key": "fti_measured", "label": "FTI measured (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "fti", "label": "FTI (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
            ],
        },
        {
            "name": "Anemia",
            "model_file": os.path.join(MODELS_DIR, "Anemia_RF.pkl"),
            "fields": [
                {"key": "gender", "label": "Gender", "type": "select", "options": ["Male", "Female"], "value": "Female"},
                {"key": "hemoglobin", "label": "Hemoglobin (g/dL)", "type": "number", "min": 0.0, "max": 30.0, "value": 12.0, "step": 0.1},
                {"key": "mch", "label": "MCH", "type": "number", "min": 0.0, "max": 200.0, "value": 28.0, "step": 0.1},
                {"key": "mchc", "label": "MCHC", "type": "number", "min": 0.0, "max": 50.0, "value": 33.0, "step": 0.1},
                {"key": "mcv", "label": "MCV", "type": "number", "min": 0.0, "max": 200.0, "value": 85.0, "step": 0.1},
            ],
        },
        {
            "name": "Osteoporosis",
            "model_file": os.path.join(MODELS_DIR, "Osteoporosis_RF.pkl"),
            "fields": [
                {"key": "age", "label": "Age", "type": "number", "min": 10.0, "max": 120.0, "value": 60.0, "step": 1.0},
                {"key": "gender", "label": "Gender", "type": "select", "options": ["Male", "Female"], "value": "Female"},
                {"key": "hormonal_changes", "label": "Hormonal changes (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "Yes"},
                {"key": "family_history", "label": "Family history (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "race_ethnicity", "label": "Race/Ethnicity", "type": "select", "options": ["Indian", "Other"], "value": "Indian"},
                {"key": "body_weight", "label": "Body weight (kg)", "type": "number", "min": 20.0, "max": 300.0, "value": 60.0, "step": 0.1},
                {"key": "calcium_intake", "label": "Calcium intake (mg/day)", "type": "number", "min": 0.0, "max": 5000.0, "value": 800.0, "step": 1.0},
                {"key": "vitamin_d_intake", "label": "Vitamin D intake (IU/day)", "type": "number", "min": 0.0, "max": 10000.0, "value": 400.0, "step": 1.0},
                {"key": "physical_activity", "label": "Physical activity (Low/Medium/High)", "type": "select", "options": ["Low", "Medium", "High"], "value": "Medium"},
                {"key": "smoking", "label": "Smoking (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
                {"key": "alcohol", "label": "Alcohol consumption (None/Moderate/High)", "type": "select", "options": ["None", "Moderate", "High"], "value": "None"},
                {"key": "medical_conditions", "label": "Medical conditions (comma separated)", "type": "text", "value": ""},
                {"key": "medications", "label": "Medications (comma separated)", "type": "text", "value": ""},
                {"key": "prior_fractures", "label": "Prior fractures (Yes/No)", "type": "select", "options": ["No", "Yes"], "value": "No"},
            ],
        },
    ]

    for i, (tab, ddef) in enumerate(zip(tabs, disease_defs)):
        with tab:
            st.subheader(ddef["name"])
          
            form_key = f"form_{i}"
            with st.form(form_key):
                inputs = {}
                # build inputs
                for f in ddef["fields"]:
                    if f["type"] == "number":
                        # Force float-typed numeric arguments (Streamlit requires consistent numeric types)
                        min_val = float(f.get("min", 0.0))
                        max_val = float(f.get("max", 1e6))
                        default_val = float(f.get("value", 0.0))
                        step = float(f.get("step", 1.0))
                        inputs[f["key"]] = st.number_input(
                            f["label"],
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            step=step,
                            key=f"{form_key}_{f['key']}"
                        )
                    elif f["type"] == "select":
                        inputs[f["key"]] = st.selectbox(
                            f["label"],
                            f["options"],
                            index=f["options"].index(f.get("value", f["options"][0])),
                            key=f"{form_key}_{f['key']}"
                        )
                    else:  # text
                        inputs[f["key"]] = st.text_input(
                            f["label"],
                            value=f.get("value", ""),
                            key=f"{form_key}_{f['key']}"
                        )

                patient_ref = st.text_input("Patient ID / Name (optional)", key=f"{form_key}_patient")
                submitted = st.form_submit_button("Predict")

            # handle submission outside the with-block
            if submitted:
                # Build feature vector in declared order. Convert Yes/No to 1/0, genders if needed.
                feature_vector = []
                for f in ddef["fields"]:
                    v = inputs[f["key"]]
                    # standard conversions
                    if isinstance(v, str) and v.lower() in ("yes", "no"):
                        v = 1 if v.lower() == "yes" else 0
                    # Map gender strings to numeric if model expects numbers (common). Adjust if your model expects different encoding.
                    if isinstance(v, str) and v in ("Male", "Female"):
                        # female=1, male=0 (change if your model is different)
                        v = 1 if v == "Female" else 0
                    # For sex 'M'/'F' used in Thyroid:
                    if isinstance(v, str) and v in ("M", "F"):
                        v = 0 if v == "M" else 1
                    # numeric values from st.number_input come as floats; convert integers where appropriate
                    if isinstance(v, float) and v.is_integer():
                        # keep as int to match some models that were trained on ints
                        v = int(v)
                    feature_vector.append(v)

                # load model: uploaded overrides local
                    model = load_model(ddef["model_file"])

                result = predict_with_model(model, feature_vector)

                st.write("**Prediction result**")
                if result.get("label") == "Error":
                    st.error(f"Model error: {result.get('error')}")
                else:
                    label = result.get("label")
                    prob = result.get("probability")

                    # user facing messages
                    if label == "Positive":
                        st.success(f"This person **has {ddef['name']}**")
                    elif label == "Negative":
                        st.info(f"This person **does not have {ddef['name']}**")
                    else:
                        st.write(f"Prediction: {label}")

                    # details
                    st.write(f"**Model label:** {label}")
                    if prob is not None:
                        st.write(f"**Predicted probability (positive):** {prob:.3f}")
                    else:
                        st.write("Probability not available for this model.")

                # save prediction record (audit)
                record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "disease": ddef["name"],
                    "patient": patient_ref,
                    "features": str(feature_vector),
                    "label": result.get("label"),
                    "probability": result.get("probability"),
                }
                preds_file = os.path.join(DATA_DIR, "predictions.csv")
                dfrec = pd.DataFrame([record])
                if os.path.exists(preds_file):
                    dfrec.to_csv(preds_file, mode="a", header=False, index=False)
                else:
                    dfrec.to_csv(preds_file, index=False)

    st.markdown("---")
    if st.button("⬅️ Back to Home "):
        st.session_state.page = "home"


# -----------------------
# App routing
# -----------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    landing_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.page == "detect":
    detect_page()

