# app.py
"""
SheHealth - Multi-disease Streamlit app (safe feature alignment)
- Prevents 'X has N features but model expects M' errors by aligning inputs to model expectations.
- Displays friendly messages and debug info in the UI.
"""

import os
from datetime import datetime
import pickle
import joblib

import streamlit as st
import pandas as pd

# Paths
MODELS_DIR = "models"
DATA_DIR = "data"
PATIENTS_FILE = os.path.join(DATA_DIR, "patients.csv")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="SheHealth", layout="wide")


# -----------------------
# Utilities
# -----------------------
def load_model(path):
    """Try to load a model. Accepts joblib/pickle.
    Also allow joblib.dump((model, feature_names), ...) pattern and returns (model, feature_names) if present."""
    if not path:
        return None, None
    if not os.path.exists(path):
        return None, None
    try:
        loaded = joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load model {path}: {e}")
            return None, None

    # Support case model saved together with feature names: (model, feature_names)
    if isinstance(loaded, (list, tuple)) and len(loaded) == 2:
        model_obj, feature_names = loaded
        return model_obj, list(feature_names)
    else:
        # single model object
        return loaded, None


def model_expected_info(model, feature_names_from_file=None):
    """Return (expected_names, expected_count) if available."""
    if model is None:
        return None, None
    # scikit-learn sets feature_names_in_ (np.array) and n_features_in_
    expected_names = None
    expected_count = None
    try:
        expected_names = list(getattr(model, "feature_names_in_", None)) if getattr(model, "feature_names_in_", None) is not None else None
    except Exception:
        expected_names = None

    try:
        expected_count = int(getattr(model, "n_features_in_", None)) if getattr(model, "n_features_in_", None) is not None else None
    except Exception:
        expected_count = None

    # If caller provided feature_names_from_file (joblib saved tuple), prefer that list
    if feature_names_from_file:
        expected_names = feature_names_from_file
        expected_count = len(feature_names_from_file)

    return expected_names, expected_count


def safe_predict(model, X_vec):
    """Predict and return dict with label/probability or error.
    X_vec is a list of numbers for a single sample."""
    if model is None:
        # Dummy: app-internal test behavior
        return {"label": "Negative", "probability": 0.2}

    try:
        arr = [list(map(float, X_vec))]
    except Exception as e:
        return {"label": "Error", "error": f"Failed to coerce features to floats: {e}"}

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(arr)[0]
            # try to interpret positive class as probs[1] if shape allows
            prob_pos = float(probs[1]) if len(probs) > 1 else float(max(probs))
            label = "Positive" if prob_pos >= 0.5 else "Negative"
            return {"label": label, "probability": prob_pos}
        else:
            pred = model.predict(arr)[0]
            # interpret numeric or string predictions
            if isinstance(pred, (int, float)):
                label = "Positive" if pred == 1 else "Negative"
            else:
                label = "Positive" if str(pred).strip().lower() in ("positive", "pos", "1", "true", "yes", "y") else "Negative"
            return {"label": label, "probability": None}
    except Exception as e:
        return {"label": "Error", "error": str(e)}


def append_patient(row: dict):
    """
    Append a patient row (dict) to patients csv.
    Mark row as manually entered so the landing page only shows manual entries.
    """
    # Ensure we always write an 'is_manual' flag for manual saves
    row = dict(row)  # copy to avoid mutating caller
    row["is_manual"] = True

    df = pd.DataFrame([row])
    if os.path.exists(PATIENTS_FILE):
        df.to_csv(PATIENTS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(PATIENTS_FILE, index=False)


def read_patients():
     if os.path.exists(PATIENTS_FILE):
        try:
            return pd.read_csv(PATIENTS_FILE)
        except Exception:
            return pd.DataFrame()
            

def save_prediction_record(record: dict):
    df = pd.DataFrame([record])
    if os.path.exists(PREDICTIONS_FILE):
        df.to_csv(PREDICTIONS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(PREDICTIONS_FILE, index=False)


def display_prediction_ui(ddef, result):
    """
    Show user-facing messages for a prediction result.
    ddef: disease def dict (needs ddef['name'])
    result: dict from safe_predict: {'label':..., 'probability':...} or {'label':'Error','error':...}
    """
    if result is None:
        st.error("No prediction result available.")
        return

    if result.get("label") == "Error":
        st.error(f"Model error: {result.get('error')}")
        return

    label = result.get("label")
    prob = result.get("probability")

    label_str = str(label).strip()
    label_low = label_str.lower()

    positive_tokens = {"positive", "pos", "yes", "y", "1", "true", "t", "disease", "present"}
    negative_tokens = {"negative", "neg", "no", "n", "0", "false", "f", "healthy", "none", "absent"}

    if label_low in positive_tokens:
        st.success(f"This person **has {ddef['name']}**")
    elif label_low in negative_tokens:
        st.info(f"This person **does not have {ddef['name']}**")
    else:
        if prob is not None:
            try:
                p = float(prob)
                if p >= 0.5:
                    st.success(f"This person **has {ddef['name']}** (prob={p:.2f})")
                else:
                    st.info(f"This person **does not have {ddef['name']}** (prob={p:.2f})")
            except Exception:
                st.write(f"Prediction: {label_str}")
        else:
            st.write(f"Prediction: {label_str}")

    # transparency
    st.write("**Model label:**", label_str)
    if prob is not None:
        try:
            st.write("**Predicted probability (positive):**", f"{float(prob):.3f}")
        except Exception:
            st.write("**Predicted probability (positive):**", prob)


# -----------------------
# UI pages
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
        if st.button("📝 Register patient data", key="register_btn"):
            st.session_state.page = "register"
    with col2:
        if st.button("🩺 Detect disease", key="detect_btn"):
            st.session_state.page = "detect"

    st.markdown("---")
    st.subheader("Available saved patients")

    patients = read_patients()

    # Filter to manual entries only:
    if not patients.empty:
        if "is_manual" in patients.columns:
            # Keep only rows where is_manual == True
            manual_patients = patients[patients["is_manual"] == True]
        else:
            # Backwards-compatible fallback:
            # Show only rows with a meaningful 'name' (not NaN, not 'None', not empty)
            name_col = patients.get("name")
            if name_col is None:
                manual_patients = pd.DataFrame()
            else:
                # coerce to str then filter
                manual_patients = patients[
                    name_col.astype(str).str.strip().str.lower().replace("nan", "").replace("none", "") != ""
                ]
                # above may still leave rows, but ensures 'None'/'nan' empty strings get filtered

    else:
        manual_patients = pd.DataFrame()

    if manual_patients.empty:
        st.info("No patients registered yet.")
    else:
        st.dataframe(manual_patients.tail(20))

        
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
            "age": int(age) if float(age).is_integer() else float(age),
            "gender": gender,
            "phone": phone,
            "notes": notes,
        }
        append_patient(row)
        st.success("Patient saved ✅")
        st.experimental_rerun()

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"


def detect_page():
    st.header("Detect disease")
    st.write("Choose a disease tab, enter inputs, and press Predict.")

    tabs = st.tabs(["PCOS", "Thyroid", "Anemia", "Osteoporosis"])

    # Disease definitions with fields (order matters for model when no feature names exist)
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

    # Build UI tabs and forms
    for i, (tab, ddef) in enumerate(zip(tabs, disease_defs)):
        with tab:
            st.subheader(ddef["name"])
            form_key = f"form_{i}"
            with st.form(form_key):
                inputs = {}
                for f in ddef["fields"]:
                    if f["type"] == "number":
                        min_val = float(f.get("min", 0.0))
                        max_val = float(f.get("max", 1e6))
                        default_val = float(f.get("value", 0.0))
                        step = float(f.get("step", 1.0))
                        inputs[f["key"]] = st.number_input(
                            f["label"], min_value=min_val, max_value=max_val,
                            value=default_val, step=step, key=f"{form_key}_{f['key']}"
                        )
                    elif f["type"] == "select":
                        opts = f.get("options", [])
                        default = f.get("value", opts[0] if opts else None)
                        try:
                            idx = opts.index(default) if default in opts else 0
                        except Exception:
                            idx = 0
                        inputs[f["key"]] = st.selectbox(
                            f["label"], opts, index=idx, key=f"{form_key}_{f['key']}"
                        )
                    else:  # text
                        inputs[f["key"]] = st.text_input(
                            f["label"], value=f.get("value", ""), key=f"{form_key}_{f['key']}"
                        )

                patient_ref = st.text_input("Patient ID / Name (optional)", key=f"{form_key}_patient")
                submitted = st.form_submit_button("Predict")

            # handle submission
            if submitted:
                # Build feature vector in declared order & feature_names list
                feature_vector = []
                feature_names = [f["key"] for f in ddef["fields"]]

                for f in ddef["fields"]:
                    v = inputs[f["key"]]

                    # Normalize strings
                    if isinstance(v, str):
                        v_str = v.strip()
                    else:
                        v_str = v

                    # Yes/No -> 1/0
                    if isinstance(v_str, str) and v_str.lower() in ("yes", "no"):
                        v = 1 if v_str.lower() == "yes" else 0

                    # Male/Female -> 1/0 (female=1, male=0)
                    elif isinstance(v_str, str) and v_str in ("Male", "Female"):
                        v = 1 if v_str == "Female" else 0

                    # M/F -> 0/1 (M=0, F=1)
                    elif isinstance(v_str, str) and v_str in ("M", "F"):
                        v = 0 if v_str == "M" else 1

                    # Low/Medium/High ordinal mapping
                    elif isinstance(v_str, str) and v_str in ("Low", "Medium", "High"):
                        v = {"Low": 0, "Medium": 1, "High": 2}[v_str]

                    # Alcohol mapping None/Moderate/High
                    elif isinstance(v_str, str) and v_str in ("None", "Moderate", "High"):
                        v = {"None": 0, "Moderate": 1, "High": 2}[v_str]

                    # free text -> count of comma-separated items
                    elif isinstance(v_str, str) and f.get("type") == "text":
                        v = len([s for s in v_str.split(",") if s.strip()])

                    # If numeric-like float but integer-valued, keep as int
                    elif isinstance(v, float) and float(v).is_integer():
                        v = int(v)

                    # Last attempt: coerce string to float
                    if isinstance(v, str):
                        try:
                            v = float(v.replace(",", ""))
                        except Exception:
                            st.warning(f"Couldn't coerce value for '{f.get('label', f.get('key'))}' -> '{v}'. Using 0 as fallback.")
                            v = 0.0

                    # Ensure numeric value
                    try:
                        feature_vector.append(float(v))
                    except Exception:
                        # fallback safe value
                        feature_vector.append(0.0)

                # Load model (and possibly saved feature names)
                model_obj, model_feature_names = load_model(ddef["model_file"])
                expected_names, expected_count = model_expected_info(model_obj, model_feature_names)

                st.write("**Prediction result**")

                # If we have expected names -> align by name
                if expected_names is not None:
                    st.info(f"Model expects {len(expected_names)} named features. Aligning by name.")
                    provided_map = {name: val for name, val in zip(feature_names, feature_vector)}
                    # Align in expected order, missing features set to 0.0
                    X_aligned = [provided_map.get(name, 0.0) for name in expected_names]
                    X_to_pred = [float(x) for x in X_aligned]
                    
                else:
                    # No feature names; use counts
                    if expected_count is not None and len(feature_vector) != expected_count:
                        st.warning(f"Model expects {expected_count} features but you provided {len(feature_vector)}. Padding/trimming to match.")
                        if len(feature_vector) > expected_count:
                            X_to_pred = feature_vector[:expected_count]
                        else:
                            X_to_pred = feature_vector + [0.0] * (expected_count - len(feature_vector))
                        st.write("Adjusted feature vector length:", len(X_to_pred))
                    else:
                        X_to_pred = feature_vector

                # Final sanity check: ensure X_to_pred is numeric list
                try:
                    X_to_pred = [float(x) for x in X_to_pred]
                except Exception as e:
                    st.error(f"Failed to prepare numeric feature vector: {e}")
                    X_to_pred = [float(0.0)] * (expected_count or len(X_to_pred))

                # Predict
                result = safe_predict(model_obj, X_to_pred)

                # Display friendly UI using helper
                display_prediction_ui(ddef, result)

                # Save a single prediction record (audit)
                record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "disease": ddef["name"],
                    "patient": patient_ref,
                    "provided_feature_names": ",".join(feature_names),
                    "provided_feature_vector": str(feature_vector),
                    "aligned_feature_vector": str(X_to_pred),
                    "model_expected_feature_names": ",".join(expected_names) if expected_names is not None else "",
                    "model_expected_count": expected_count if expected_count is not None else "",
                    "label": result.get("label"),
                    "probability": result.get("probability"),
                }
                save_prediction_record(record)

    st.markdown("---")
    if st.button("⬅️ Back to Home"):
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
else:
    landing_page()
