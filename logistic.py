import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
      /*  header {visibility: hidden;} */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
            
            header [data-testid="stHeaderActionButtons"] button {
    color: white !important;
    fill: white !important;
}

/* Optional: make the header background dark so white icons look nice */
header {
    background: linear-gradient(90deg, #005f73, #0a9396) !important;
}
            

        .stApp {
            background: linear-gradient(120deg, #f6f9fc, #e9f0fa);
            color: #1e1e1e;
            font-family: 'Segoe UI', sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #005f73, #0a9396);
            color: white;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }

        .stNumberInput > div:hover {
            border: 1px solid #94d2bd !important;
            box-shadow: 0 0 10px rgba(148, 210, 189, 0.5);
            transition: 0.3s;
        }

        div.stButton > button {
            background: linear-gradient(90deg, #0a9396, #94d2bd);
            color: white;
            border: none;
            border-radius: 12px;
            height: 50px;
            font-size: 18px;
            font-weight: 600;
            transition: 0.3s ease-in-out;
        }

        div.stButton > button:hover {
            transform: scale(1.03);
            box-shadow: 0 0 15px rgba(10,147,150,0.3);
        }

        .prediction-card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0px 5px 20px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .prediction-card:hover {
            transform: translateY(-5px);
            box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
        }

        .footer {
            text-align: center;
            font-size: 0.9rem;
            color: #666;
        }
    </style>
""", unsafe_allow_html=True)

model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🩺 Breast Cancer Prediction App")
st.markdown("""
Welcome to the **Breast Cancer Prediction System**.  
Enter tumor measurements on the left to predict whether a tumor is:
- 🧬 **Malignant (Cancerous)**  
- 🩵 **Benign (Non-Cancerous)**
""")

st.divider()

st.sidebar.header("Enter Tumor Measurements:")

features = [
    'radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
    'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'smoothness_se', 'compactness_se',
    'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

user_data = []

for feature in features:
    value = st.sidebar.number_input(
        f"{feature.replace('_', ' ').title()}:",
        min_value=0.0,
        step=0.0001,
        format="%.5f",
        key=feature
    )
    user_data.append(value)

input_data = pd.DataFrame([user_data], columns=features)
input_scaled = scaler.transform(input_data)

st.subheader("🧩 Model Output")
st.markdown("Click below to predict the tumor type:")

if st.button("🔍 Predict", use_container_width=True):
    prediction = model.predict(input_scaled)[0]
    result = "🧬 **Malignant (Cancerous)**" if prediction == 1 else "🩵 **Benign (Non-Cancerous)**"
    st.markdown(f"""
    <div class="prediction-card">
        <h3 style='text-align:center;'>🎯 Prediction Result</h3>
        <h2 style='text-align:center; color:{'#ae2012' if prediction==1 else '#0a9396'};'>{result}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
