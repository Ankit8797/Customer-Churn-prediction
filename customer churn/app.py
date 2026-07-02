import streamlit as st
import pandas as pd
import pickle
import numpy as np
from preprocessing import load_data, preprocess_data
from evaluation import load_model_and_encoders, evaluate_model, plot_confusion_matrix, plot_roc_curve, get_feature_importance
import eda
import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📊 Telco Customer Churn Prediction")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Predict Churn", "EDA Dashboard", "Model Performance"])

# Load model and encoders
model, encoders = load_model_and_encoders()

# Main pages
if page == "Predict Churn":
    st.header("Enter Customer Details")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=1)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=70.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=0.0)
        
        submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        # Build input dataframe
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        # Preprocess using saved encoders
        X_input, _ = preprocess_data(input_data, fit_encoders=False, encoders=encoders)
        
        # Predict
        proba = model.predict_proba(X_input)[0, 1]
        pred = model.predict(X_input)[0]
        
        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"⚠️ Customer is likely to churn (probability: {proba:.2%})")
        else:
            st.success(f"✅ Customer is likely to stay (churn probability: {proba:.2%})")

elif page == "EDA Dashboard":
    st.header("Exploratory Data Analysis")
    # Load data
    df = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    st.subheader("Churn Distribution")
    fig = eda.plot_churn_distribution(df)
    st.pyplot(fig)
    
    st.subheader("Tenure vs Churn")
    fig = eda.plot_tenure_by_churn(df)
    st.pyplot(fig)
    
    st.subheader("Monthly Charges vs Churn")
    fig = eda.plot_monthly_charges_by_churn(df)
    st.pyplot(fig)
    
    st.subheader("Churn Rate by Contract Type")
    fig = eda.plot_categorical_churn(df, 'Contract')
    st.pyplot(fig)
    
    st.subheader("Churn Rate by Internet Service")
    fig = eda.plot_categorical_churn(df, 'InternetService')
    st.pyplot(fig)
    
    st.subheader("Churn Rate by Payment Method")
    fig = eda.plot_categorical_churn(df, 'PaymentMethod')
    st.pyplot(fig)
    
    st.subheader("Contract × Tenure Heatmap")
    fig = eda.plot_contract_tenure_heatmap(df)
    st.pyplot(fig)

else:  # Model Performance
    st.header("Model Performance Evaluation")
    
    # Load test data from training (we need to have test set saved; we'll regenerate)
    # For simplicity, we'll reload data and split on the fly
    from sklearn.model_selection import train_test_split
    df = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    X, y = preprocess_data(
    df,
    fit_encoders=False,
    encoders=encoders
)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    metrics, y_pred, y_proba = evaluate_model(X_test, y_test)
    
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    col3.metric("F1 Score (Churn)", f"{metrics['classification_report']['1']['f1-score']:.3f}")
    
    st.subheader("Confusion Matrix")
    fig = plot_confusion_matrix(metrics['confusion_matrix'])
    st.pyplot(fig)
    
    st.subheader("ROC Curve")
    fig = plot_roc_curve(y_test, y_proba)
    st.pyplot(fig)
    
    st.subheader("Feature Importance")
    feature_names = X.columns.tolist()
    fig = get_feature_importance(model, feature_names)
    st.pyplot(fig)
    
    st.subheader("Classification Report")
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    st.dataframe(report_df)