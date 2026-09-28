import streamlit as st
import joblib
import pandas as pd 
import numpy as np

# Load trained model
artifact = joblib.load("models/credit_risk_model.pkl")

model = artifact["model"]
threshold = artifact["threshold"]

st.title("Credit Risk Prediction")

st.write(f"Classification threshold: {threshold:.2f}")


st.header("Borrower Information") 
investment_total = st.number_input( "Investment Total", min_value=0.0, value=10000.0 ) 
current_balance = st.number_input( "Current Account Balance", min_value=0.0, value=5000.0 ) 
installment_size = st.number_input( "Installment Size", min_value=0.0, value=0.0 )
due_payment = st.number_input( "Due Payment", min_value=0.0, value=0.0 ) 
marital_status = st.selectbox( "Marital Status", ["M", "U", "O"] ) 
gender = st.selectbox( "Gender", ["M", "F", "O"] ) 
client_type = st.selectbox( "Client Type", ["Urban", "Rural", "Semi-urban", "UNKNOWN"] )


has_due_payment = int(due_payment > 0) 
has_installment = int(installment_size > 0) 
if investment_total > 0: 
    balance_to_investment = ( current_balance / investment_total ) 
    installment_to_investment = ( installment_size / investment_total )
else: 
    balance_to_investment = 0 
    installment_to_investment = 0 

log_balance_to_investment = np.log1p( balance_to_investment ) 
log_installment_to_investment = np.log1p( installment_to_investment )

input_data = pd.DataFrame({ "INVESTMENT_TOTAL": [investment_total], "ACCCURRENTBALANCE": [current_balance], "INSTALL_SIZE": [installment_size], "DUE_PAYMENT": [due_payment], 
                           "HAS_DUE_PAYMENT": [has_due_payment], "HAS_INSTALLMENT": [has_installment], "BALANCE_TO_INVESTMENT": [balance_to_investment], 
                           "LOG_BALANCE_TO_INVESTMENT": [log_balance_to_investment], "INSTALLMENT_TO_INVESTMENT": [installment_to_investment], 
                           "LOG_INSTALLMENT_TO_INVESTMENT": [log_installment_to_investment], "INF_MARITAL_STATUS": [marital_status], "INF_GENDER": [gender],
                             "CLIENT_TYPE": [client_type] })

st.subheader("Model Input") 
st.dataframe(input_data)

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Credit Risk"):

    probability_bad = model.predict_proba(input_data)[0, 1]

    prediction = ("Bad" if probability_bad >= threshold else "Good")

    st.subheader("Prediction")

    st.metric('Probability of Bad Loan',f" {probability_bad:.2%}")

    if prediction == "Bad": st.error("Classification: Bad") 
    else: st.success("Classification: Good") 
    # Explain threshold 
    st.write( f"The model classifies the borrower as **{prediction}** " f"using a threshold of **{threshold:.2f}**." )