import streamlit as st
import pickle
import numpy as np
import xgboost

# Load models and scalers
with open('E:\DSDemo\env\parkinsons_model1.pkl', 'rb') as f:
    parkinsons_data = pickle.load(f)
    parkinsons_scaler = parkinsons_data['scaler']
    parkinsons_model = parkinsons_data['model1']

with open('E:\DSDemo\env\india_liver_model1.pkl', 'rb') as f:
    liver_data = pickle.load(f)
    liver_encoder = liver_data['scaler']
    liver_model = liver_data['model1']

with open('E:\DSDemo\env\kidney_model6.pkl', 'rb') as f:
    kidney_data = pickle.load(f)
    kidney_encoder = kidney_data['scaler']
    kidney_model = kidney_data['model6']

# Streamlit dashboard
st.header("MULTIPLE DISEASE PREDICTION")
st.subheader("Disease Prediction Dashboard")
selected_dataset = st.sidebar.selectbox(
    "Select a Disease to Predict",
    ("Parkinson's Disease","Liver Disease","Kidney Disease")
)

if selected_dataset == "Parkinson's Disease":
    st.header("Parkinson's Disease Prediction")
    MDVP_Fo = st.number_input("MDVP:Fo(Hz) (Average Vocal Fundamental Frequency)", min_value=0.0, step=0.1)
    MDVP_Fhi = st.number_input("MDVP:Fhi(Hz) (Maximum Vocal Fundamental Frequency)", min_value=0.0, step=0.1)
    MDVP_Flo = st.number_input("MDVP:Flo(Hz) (Minimum Vocal Fundamental Frequency)", min_value=0.0, step=0.1)
    MDVP_Jitter = st.number_input("MDVP:Jitter(%) (Jitter Percent)", min_value=0.0, step=0.001)
    MDVP_Shimmer = st.number_input("MDVP:Shimmer (Shimmer dB)", min_value=0.0, step=0.001)
    HNR = st.number_input("HNR (Harmonic-to-Noise Ratio)", min_value=0.0, step=0.1)
    RPDE = st.number_input("RPDE (Recurrence Period Density Entropy)", min_value=0.0, step=0.001)
    DFA = st.number_input("DFA (Detrended Fluctuation Analysis)", min_value=0.0, step=0.001)
    spread1 = st.number_input("Spread1 (Vocal Fundamental Frequency Variation)", step=0.001)
    spread2 = st.number_input("Spread2 (Second Variation Measure)", step=0.001)
    D2 = st.number_input("D2 (Signal Dynamical Complexity)", min_value=0.0, step=0.001)
    PPE = st.number_input("PPE (Pitch Period Entropy)", min_value=0.0, step=0.001)

    input_features = np.array([[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter,
                                 MDVP_Shimmer, HNR, RPDE, DFA, spread1, spread2, D2, PPE 
    ]])
    scaled_features = parkinsons_scaler.transform(input_features)
    if st.button("Predict"):
        prediction = parkinsons_model.predict(scaled_features)
        result = "Positive for Parkinson's Disease" if prediction[0] == 1 else "Negative for Parkinson's Disease"
        st.write(f"Prediction: {result}")


elif selected_dataset == "Liver Disease":
    st.header("Liver Disease Prediction")
    Age = st.number_input("Age", min_value=1, step=1)
    Total_Bilirubin = st.number_input("Total Bilirubin", min_value=0.0)
    Direct_Bilirubin = st.number_input("Direct Bilirubin", min_value=0.0)
    Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase", min_value=0.0)
    Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase", min_value=0.0)
    Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0.0)
    Total_Proteins = st.number_input("Total Proteins", min_value=0.0)
    Albumin = st.number_input("Albumin", min_value=0.0)
    Albumin_and_Globulin_Ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0)


    input_features = np.array([[
        Age,Total_Bilirubin, Direct_Bilirubin,
        Alkaline_Phosphotase, Alamine_Aminotransferase,
        Aspartate_Aminotransferase, Total_Proteins, Albumin,
        Albumin_and_Globulin_Ratio
    ]])

    try:
        scaled_features = liver_encoder.transform(input_features)

        # Predict button
        if st.button("Predict"):
            prediction = liver_model.predict(scaled_features)
            result = "Positive for Liver Disease" if prediction[0] == 1 else "Negative for Liver Disease"
            st.write(f"Prediction: {result}")
    except ValueError as e:
        st.error(f"Error during prediction: {str(e)}")

    
elif selected_dataset == "Kidney Disease":
    st.header("Kidney Disease Prediction")
    age = st.number_input("Age", min_value=1, step=1)
    bp = st.number_input("Blood Pressure", min_value=0)
    sg = st.number_input("Specific Gravity", min_value=1.0, max_value=1.030, step=0.001)
    al = st.number_input("Albumin", min_value=0.0)
    su = st.number_input("Sugar", min_value=0.0)
    rbc= st.number_input("Red Blood Cells", min_value=0.0)
    pc = st.number_input("Pus Cells", min_value=0.0)
    sc = st.number_input("Serum Creatinine Level", min_value=0.0)
    sod= st.number_input("Sodium Level in Blood", min_value=0.0)
    wc = st.number_input("White Blood Cells", min_value=0.0)
    rc = st.number_input("Red Blood Cell Count", min_value=0.0)
    cad= st.number_input("Coronary artery Disease", min_value=0.0)
    appet= st.number_input("Appetite Status", min_value=0.0)


    input_features = np.array([[age,bp,sg,al,su,rbc,pc,sc,sod,wc,rc,cad,appet]])
    scaled_features = kidney_encoder.transform(input_features)
    if st.button("Predict"):
        prediction = kidney_model.predict(scaled_features)
        result = "Positive for Kidney Disease" if prediction[0] == 1 else "Negative for Kidney Disease"
        st.write(f"Prediction: {result}")





