# app.py (cleaned: no debug output, back-home buttons, stable keys)
import os
from datetime import datetime
import pickle
import joblib
import streamlit as st
import pandas as pd

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="SheHealth", layout="centered")
MODELS_DIR = "models"
DATA_DIR = "data"
PATIENTS_FILE = os.path.join(DATA_DIR, "patients.csv")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def load_model(path):
    """Try joblib -> pickle. Return (model, feature_names_list_or_None)."""
    if not path or not os.path.exists(path):
        return None, None
    try:
        loaded = joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load model '{path}': {e}")
            return None, None

    if isinstance(loaded, (list, tuple)) and len(loaded) == 2:
        return loaded[0], list(loaded[1])
    return loaded, None


def model_info(model, feature_names_from_file=None):
    """Return (expected_names_or_None, expected_count_or_None)."""
    if model is None:
        return None, None
    expected_names = None
    expected_count = None
    try:
        if hasattr(model, "feature_names_in_"):
            expected_names = list(getattr(model, "feature_names_in_"))
    except Exception:
        expected_names = None
    try:
        if hasattr(model, "n_features_in_"):
            expected_count = int(getattr(model, "n_features_in_"))
    except Exception:
        expected_count = None

    if feature_names_from_file:
        expected_names = feature_names_from_file
        expected_count = len(feature_names_from_file)

    return expected_names, expected_count


def safe_predict(model, X):
    """Return dict: {label, probability (maybe None), error (maybe None)}."""
    if model is None:
        return {"label": "Negative (no model)", "probability": 0.0}

    try:
        arr = [list(map(float, X))]
    except Exception as e:
        return {"label": None, "probability": None, "error": f"Feature conversion error: {e}"}

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(arr)[0]
            if len(probs) >= 2:
                prob_pos = float(probs[1])
            else:
                prob_pos = float(max(probs))
            label = "Positive" if prob_pos >= 0.5 else "Negative"
            return {"label": label, "probability": prob_pos}
        else:
            pred = model.predict(arr)[0]
            if isinstance(pred, (int, float)):
                label = "Positive" if float(pred) == 1.0 else "Negative"
            else:
                s = str(pred).strip().lower()
                label = "Positive" if s in ("positive", "pos", "1", "true", "yes") else "Negative"
            return {"label": label, "probability": None}
    except Exception as e:
        # final fallback: return error so UI can show it
        return {"label": None, "probability": None, "error": f"Model prediction error: {e}"}


def append_csv(path, row: dict):
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def load_patients_df():
    if os.path.exists(PATIENTS_FILE):
        try:
            return pd.read_csv(PATIENTS_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# -------------------------
# UI - Navigation helpers
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"


def go_home():
    st.session_state.page = "home"
    st.rerun()


# -------------------------
# Simple disease definitions
# -------------------------
DISEASES = [
    {
        "name": "PCOS",
        "model": os.path.join(MODELS_DIR, "PCOS_RF.pkl"),
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
        "model": os.path.join(MODELS_DIR, "Thyroid_RF.pkl"),
        "fields": [
            {"key": "age", "label": "Age", "type": "number", "min": 0.0, "max": 120.0, "value": 25.0, "step": 1.0},
            {"key": "sex", "label": "Sex", "type": "select", "options": ["Female", "Other"], "value": "Female"},
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
        "model": os.path.join(MODELS_DIR, "Anemia_RF.pkl"),
        "fields": [
            {"key": "gender", "label": "Gender", "type": "select", "options": ["Female", "Other"], "value": "Female"},
            {"key": "hemoglobin", "label": "Hemoglobin (g/dL)", "type": "number", "min": 0.0, "max": 30.0, "value": 12.0, "step": 0.1},
            {"key": "mch", "label": "MCH", "type": "number", "min": 0.0, "max": 200.0, "value": 28.0, "step": 0.1},
            {"key": "mchc", "label": "MCHC", "type": "number", "min": 0.0, "max": 50.0, "value": 33.0, "step": 0.1},
            {"key": "mcv", "label": "MCV", "type": "number", "min": 0.0, "max": 200.0, "value": 85.0, "step": 0.1},
        ],
    },
    {
        "name": "Osteoporosis",
        "model": os.path.join(MODELS_DIR, "Osteoporosis_RF.pkl"),
        "fields": [
            {"key": "age", "label": "Age", "type": "number", "min": 10.0, "max": 120.0, "value": 25.0, "step": 1.0},
            {"key": "gender", "label": "Gender", "type": "select", "options": ["Female", "Other"], "value": "Female"},
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

# -------------------------
# Pages
# -------------------------
def home_page():
    st.title("🔬 SheHealth-Disease Prediction for Female Health Disorders")
    st.write("Choose an action below.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📝 Register patient", key="btn_register"):
            st.session_state.page = "register"
            st.rerun()
    with c2:
        if st.button("🩺 Detect disease", key="btn_detect"):
            st.session_state.page = "detect"
            st.rerun()

    df = load_patients_df()
    if not df.empty:
        st.markdown("---")
        st.subheader("Recent Patients")
        st.dataframe(df.tail(5))


def register_page():
    st.header("Register Patient")

    with st.form("reg_form", clear_on_submit=True):
        name = st.text_input("Full name")
        age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)
        gender = st.selectbox("Gender", ["Female", "Other"])
        phone = st.text_input("Phone (optional)")

        notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Save", key="save_patient")

    if submitted:
        row = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "age": age,
            "gender": gender,
            "phone": phone,
            "notes": notes,
        }
        append_csv(PATIENTS_FILE, row)
        st.success(f"✅ Patient details saved successfully!")


    # bottom back-home button
    st.markdown("---")
    if st.button("⬅️ Back to Home", key="back_home_register_bottom"):
        go_home()


def detect_page():
    st.header("Detect disease")
    tabs = st.tabs([d["name"] for d in DISEASES])

    for i, d in enumerate(DISEASES):
        with tabs[i]:
            st.subheader(d["name"])
            form_key = f"form_{d['name']}"
            with st.form(form_key, clear_on_submit=False):
                inputs = {}
                for f in d["fields"]:
                    if f["type"] == "number":
                        inputs[f["key"]] = st.number_input(
                            f["label"],
                            min_value=float(f.get("min", -1e9)),
                            max_value=float(f.get("max", 1e9)),
                            value=float(f.get("value", 0.0)),
                            step=float(f.get("step", 1.0)),
                            key=f"{form_key}_{f['key']}",
                        )
                    elif f["type"] == "select":
                        opts = f.get("options", [])
                        default = f.get("value", opts[0] if opts else None)
                        try:
                            idx = opts.index(default) if default in opts else 0
                        except Exception:
                            idx = 0
                        inputs[f["key"]] = st.selectbox(f["label"], opts, index=idx, key=f"{form_key}_{f['key']}")
                    elif f["type"] == "text":
                        inputs[f["key"]] = st.text_input(f["label"], value=f.get("value", ""), key=f"{form_key}_{f['key']}")
                    else:
                        inputs[f["key"]] = st.text_input(f["label"], value=f.get("value", ""), key=f"{form_key}_{f['key']}")

                patient_ref = st.text_input("Patient ID / Name (optional)", key=f"{form_key}_patient")
                predict_btn = st.form_submit_button("Predict", key=f"predict_{d['name']}")

            if predict_btn:
                # Build feature vector in declared order
                feature_names = [fld["key"] for fld in d["fields"]]
                feature_vector = []
                for fn in feature_names:
                    v = inputs[fn]
                    if isinstance(v, str):
                        s = v.strip()
                        if s.lower() in ("yes", "no"):
                            v = 1.0 if s.lower() == "yes" else 0.0
                        elif s in ("Male", "Female"):
                            v = 1.0 if s == "Female" else 0.0
                        elif s in ("M", "F"):
                            v = 0.0 if s == "M" else 1.0
                        else:
                            try:
                                v = float(s.replace(",", ""))
                            except Exception:
                                v = 0.0
                    try:
                        feature_vector.append(float(v))
                    except Exception:
                        feature_vector.append(0.0)

                # Load model and align features if possible
                model_obj, model_feature_names = load_model(d["model"])
                expected_names, expected_count = model_info(model_obj, model_feature_names)

                if expected_names is not None:
                    provided_map = {name: val for name, val in zip(feature_names, feature_vector)}
                    X_to_send = [provided_map.get(name, 0.0) for name in expected_names]
                elif expected_count is not None:
                    if len(feature_vector) != expected_count:
                        if len(feature_vector) > expected_count:
                            X_to_send = feature_vector[:expected_count]
                        else:
                            X_to_send = feature_vector + [0.0] * (expected_count - len(feature_vector))
                    else:
                        X_to_send = feature_vector
                else:
                    X_to_send = feature_vector

                # Ensure numeric list
                try:
                    X_to_send = [float(x) for x in X_to_send]
                except Exception:
                    X_to_send = [0.0] * len(X_to_send)

                # Predict safely
                result = safe_predict(model_obj, X_to_send)
                if result.get("error"):
                    st.error("Prediction error: " + str(result.get("error")))
                else:
                    label = result.get("label")
                    prob = result.get("probability")
                    # ONLY show prediction (no debug)
                    if prob is not None:
                        st.markdown(f"**Prediction:** {label} (probability positive = {prob:.3f})")
                        if prob >= 0.5:
                            st.success(f"Interpretation: POSITIVE for {d['name']}")
                        else:
                            st.info(f"Interpretation: NEGATIVE for {d['name']}")
                    else:
                        st.markdown(f"**Prediction:** {label}")
                        if label and "positive" in str(label).lower():
                            st.success(f"Interpretation: POSITIVE for {d['name']}")
                        else:
                            st.info(f"Interpretation: NEGATIVE for {d['name']}")

                # Save prediction record for audit (kept but not printed)
                rec = {
                    "timestamp": datetime.now().isoformat(),
                    "disease": d["name"],
                    "patient": patient_ref,
                    "provided_features": ",".join(feature_names),
                    "provided_vector": str(feature_vector),
                    "aligned_vector": str(X_to_send),
                    "model_expected_names": ",".join(expected_names) if expected_names else "",
                    "model_expected_count": expected_count if expected_count else "",
                    "label": result.get("label"),
                    "probability": result.get("probability"),
                }
                append_csv(PREDICTIONS_FILE, rec)

            # Back Home button per tab (unique key)
            st.markdown("---")
            if st.button("⬅️ Back to Home", key=f"back_home_{d['name']}"):
                go_home()


# -------------------------
# Router
# -------------------------
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.page == "detect":
    detect_page()
else:
    home_page()
