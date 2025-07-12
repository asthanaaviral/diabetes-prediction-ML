import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto"
)

dark_theme = """
    <style>
    body {
        background-color: #0e1117;
        color: #f5f6fa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.2em;
    }
    .stNumberInput input {
        background-color: #1c1c1e;
        color: white;
    }
    .stTextInput input {
        background-color: #1c1c1e;
        color: white;
    }
    </style>
"""
st.markdown(dark_theme, unsafe_allow_html=True)

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")


# st.sidebar.image("https://img.icons8.com/fluency/96/diabetes.png", use_container_width=True)
st.sidebar.title("Diabetes Predictor")
st.sidebar.markdown(
    """  
    Enter your health information to get an instant prediction of whether you're likely to have diabetes.

    This model was trained using the PIMA Indians Diabetes dataset and a Random Forest classifier.
    """
)


st.title("🩺 Diabetes Risk Prediction")
st.markdown("##### A smart ML-powered tool to assess diabetes likelihood based on medical metrics.")
st.divider()


with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=130, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin Level", min_value=0, max_value=850, value=80)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0, format="%.2f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    submitted = st.form_submit_button("🔍 Predict")


if submitted:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]

    st.divider()
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ The model predicts that you are **likely to have diabetes**.\n\n🧪 Confidence: `{prediction_proba[1]*100:.2f}%`")
    else:
        st.success(f"✅ The model predicts that you are **not likely to have diabetes**.\n\n🧪 Confidence: `{prediction_proba[0]*100:.2f}%`")

    st.divider()
    st.markdown(
        """
         **Note**: This prediction is based on statistical models and should not be considered a medical diagnosis.  
        Always consult a certified healthcare provider for professional guidance.
        """
    )

st.markdown("---")
st.markdown("Made with ❤️ by Aviral.")
