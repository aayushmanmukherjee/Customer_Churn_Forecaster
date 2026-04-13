import numpy as np
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# LOAD MODEL
loaded_model = pickle.load(open('model.pkl', 'rb'))

# PREDICTION FUNCTION
def churn_prediction(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = loaded_model.predict(input_df)

    if prediction[0] == 1:
        return "Customer will CHURN (Cancel their plan)"
    else:
        return "Customer will STAY (Continue their plan)"


# APP
def main():

    st.set_page_config(page_title="Churn Forecaster", layout="wide")

    st.title("Customer Churn Forecaster App")
    st.markdown("Forecast customer churn and analyze model performance")

    # INPUT SECTION
    st.header("Customer Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", ['0', '1'])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    with col2:
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])

    with col3:
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"]
        )

        tenure = st.number_input("Tenure", min_value=0)
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0)

    # INPUT DICT
    input_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    # PREDICTION
    st.header("Predict Result")

    if st.button("Forecast Churn"):
        result = churn_prediction(input_data)
        st.success(result)

    # MODEL PERFORMANCE
    st.header("Model Performance")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Classification Report")

        report = pd.DataFrame({
            "Class": ["No Churn", "Churn"],
            "Precision": [0.89, 0.57],
            "Recall": [0.80, 0.73],
            "F1-score": [0.84, 0.64],
            "Support": [1546, 567]
        })

        st.dataframe(report)

    with colB:
        st.subheader("Confusion Matrix")

        cm = np.array([
            [1235, 311],
            [155, 412]
        ])

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)

        st.pyplot(fig)


if __name__ == "__main__":
    main()