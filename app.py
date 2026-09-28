import streamlit as st
import joblib
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# Load trained model
artifact = joblib.load("models/credit_risk_model.pkl")
df = pd.read_csv("data/credit_risk_cleaned.csv")

model = artifact["model"]
threshold = artifact["threshold"]

metrics = artifact["metrics"]
confusion_matrix_data = artifact["confusion_matrix"]

st.title("Credit Risk Management Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Credit Risk Assessment",
        "Portfolio Analytics",
        "Model Performance"
    ]
)

if page == "Overview":

    st.header("Portfolio Overview")

    total_borrowers = len(df)
    bad_loans = (df["QUALITY_OF_LOAN"] == "B").sum()
    good_loans = (df["QUALITY_OF_LOAN"] == "G").sum()
    bad_rate = bad_loans / total_borrowers

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Borrowers",
        f"{total_borrowers:,}"
    )

    col2.metric(
        "Good Loans",
        f"{good_loans:,}"
    )

    col3.metric(
        "Bad Loans",
        f"{bad_loans:,}"
    )

    col4.metric(
        "Bad Loan Rate",
        f"{bad_rate:.2%}"
    )


elif page == "Credit Risk Assessment":

    st.header("Credit Risk Assessment")

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
    
    if st.button("Predict Credit Risk"):

        probability_bad = model.predict_proba(input_data)[0, 1]

        prediction = ("Bad" if probability_bad >= threshold else "Good")

        st.subheader("Prediction")

        st.metric('Probability of Bad Loan',f" {probability_bad:.2%}")

        if prediction == "Bad": st.error("Classification: Bad") 
        else: st.success("Classification: Good") 
        # Explain threshold 
        st.write( f"The model classifies the borrower as **{prediction}** " f"using a threshold of **{threshold:.2f}**." )



elif page == "Portfolio Analytics":

    st.header("Portfolio Analytics")
    st.caption(
        "Interactive overview of borrower characteristics, loan quality, "
        "and portfolio risk patterns."
    )

    # =========================================================
    # 1. PORTFOLIO KPIs
    # =========================================================

    total_borrowers = len(df)

    good_loans = (
        df["QUALITY_OF_LOAN"] == "G"
    ).sum()

    bad_loans = (
        df["QUALITY_OF_LOAN"] == "B"
    ).sum()

    bad_rate = bad_loans / total_borrowers

    avg_investment = df["INVESTMENT_TOTAL"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Borrowers",
        f"{total_borrowers:,}"
    )

    col2.metric(
        "Good Loans",
        f"{good_loans:,}"
    )

    col3.metric(
        "Bad Loans",
        f"{bad_loans:,}"
    )

    col4.metric(
        "Bad Loan Rate",
        f"{bad_rate:.2%}"
    )

    st.divider()


    # =========================================================
    # 2. LOAN QUALITY + CLIENT TYPE
    # =========================================================

    col1, col2 = st.columns(2)

    # ---------------------------------------------------------
    # 2A. Good vs Bad Loans
    # ---------------------------------------------------------

    with col1:

        st.subheader("Loan Quality Distribution")

        loan_quality_counts = (
            df["QUALITY_OF_LOAN"]
            .map({
                "G": "Good",
                "B": "Bad"
            })
            .value_counts()
            .reset_index()
        )

        loan_quality_counts.columns = [
            "Loan Quality",
            "Count"
        ]

        fig_quality = px.pie(
            loan_quality_counts,
            names="Loan Quality",
            values="Count",
            hole=0.55
        )

        fig_quality.update_traces(
            textinfo="percent+label"
        )

        fig_quality.update_layout(
            height=400,
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            ),
            showlegend=True
        )

        st.plotly_chart(
            fig_quality,
            use_container_width=True
        )


    # ---------------------------------------------------------
    # 2B. Bad Loan Rate by Client Type
    # ---------------------------------------------------------

    with col2:

        st.subheader("Bad Loan Rate by Client Type")

        bad_rate_by_client = (
            df.groupby("CLIENT_TYPE")["QUALITY_OF_LOAN"]
            .apply(lambda x: (x == "B").mean())
            .reset_index(name="Bad Rate")
        )

        bad_rate_by_client["Bad Rate"] *= 100

        fig_client = px.bar(
            bad_rate_by_client,
            x="CLIENT_TYPE",
            y="Bad Rate",
            text="Bad Rate",
            labels={
                "CLIENT_TYPE": "Client Type",
                "Bad Rate": "Bad Loan Rate (%)"
            }
        )

        fig_client.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig_client.update_layout(
            height=400,
            yaxis_title="Bad Loan Rate (%)",
            xaxis_title="Client Type",
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            )
        )

        st.plotly_chart(
            fig_client,
            use_container_width=True
        )


    st.divider()


    # =========================================================
    # 3. INVESTMENT RANGE ANALYSIS
    # =========================================================

    investment_bins = [
        500000,
        1000000,
        2000000,
        5000000,
        10000000,
        np.inf
    ]

    investment_labels = [
        "$500K–$1M",
        "$1M–$2M",
        "$2M–$5M",
        "$5M–$10M",
        "> $10M"
    ]

    df["INVESTMENT_RANGE"] = pd.cut(
        df["INVESTMENT_TOTAL"],
        bins=investment_bins,
        labels=investment_labels,
        include_lowest=True
    )


    # ---------------------------------------------------------
    # 3A. Loan Quality by Investment Range
    # ---------------------------------------------------------

    st.subheader("Loan Quality by Investment Range")

    loan_quality_by_investment = (
        df.groupby(
            ["INVESTMENT_RANGE", "QUALITY_OF_LOAN"],
            observed=False
        )
        .size()
        .reset_index(name="Borrowers")
    )

    loan_quality_by_investment["Loan Quality"] = (
        loan_quality_by_investment["QUALITY_OF_LOAN"]
        .map({
            "G": "Good",
            "B": "Bad"
        })
    )

    fig = px.bar(
        loan_quality_by_investment,
        x="INVESTMENT_RANGE",
        y="Borrowers",
        color="Loan Quality",
        barmode="stack",
        text="Borrowers",
        labels={
            "INVESTMENT_RANGE": "Investment Range",
            "Borrowers": "Number of Borrowers"
        }
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="inside"
    )

    fig.update_layout(
        height=450,
        xaxis_title="Investment Range",
        yaxis_title="Number of Borrowers",
        legend_title="Loan Quality"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.divider()


    # ---------------------------------------------------------
    # 3B. Bad Loan Rate by Investment Range
    # ---------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Bad Loan Rate by Investment Range")

        bad_rate_by_investment = (
            df.groupby(
                "INVESTMENT_RANGE",
                observed=False
            )["QUALITY_OF_LOAN"]
            .apply(lambda x: (x == "B").mean())
            .reset_index(name="Bad Rate")
        )

        bad_rate_by_investment["Bad Rate"] *= 100

        fig_investment_risk = px.bar(
            bad_rate_by_investment,
            x="INVESTMENT_RANGE",
            y="Bad Rate",
            text="Bad Rate",
            labels={
                "INVESTMENT_RANGE": "Investment Range",
                "Bad Rate": "Bad Loan Rate (%)"
            }
        )

        fig_investment_risk.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig_investment_risk.update_layout(
            height=450,
            yaxis_title="Bad Loan Rate (%)",
            xaxis_title="Investment Range",
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            )
        )

        st.plotly_chart(
            fig_investment_risk,
            use_container_width=True
        )


    # ---------------------------------------------------------
    # 3C. Borrower Distribution by Investment Range
    # ---------------------------------------------------------

    with col2:

        st.subheader("Borrower Distribution by Investment Range")

        investment_distribution = (
            df["INVESTMENT_RANGE"]
            .value_counts()
            .reindex(investment_labels)
            .reset_index()
        )

        investment_distribution.columns = [
            "Investment Range",
            "Borrowers"
        ]

        fig_distribution = px.bar(
            investment_distribution,
            x="Investment Range",
            y="Borrowers",
            text="Borrowers",
            labels={
                "Investment Range": "Investment Range",
                "Borrowers": "Number of Borrowers"
            }
        )

        fig_distribution.update_traces(
            texttemplate="%{text:,}",
            textposition="outside"
        )

        fig_distribution.update_layout(
            height=450,
            yaxis_title="Number of Borrowers",
            xaxis_title="Investment Range",
            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            )
        )

        st.plotly_chart(
            fig_distribution,
            use_container_width=True
        )


    st.divider()


# =========================================================
# 4. NEXT SECTION
# =========================================================

elif page == "Model Performance":

    st.header("Model Performance")

    # =========================================================
    # 1. MODEL INFORMATION
    # =========================================================

    st.subheader("Model Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.write("**Model**")
    col1.write("Random Forest")

    col2.metric(
        "Decision Threshold",
        f"{threshold:.2f}"
    )

    col3.metric(
        "ROC-AUC",
        "0.766"
    )

    col4.metric(
        "PR-AUC",
        "0.394"
    )


    st.divider()


    # =========================================================
    # 2. TEST SET PERFORMANCE
    # =========================================================

    st.subheader("Test Set Performance")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.3f}"
    )

    col2.metric(
        "Precision (Bad)",
        f"{metrics['precision']:.3f}"
    )

    col3.metric(
        "Recall (Bad)",
        f"{metrics['recall']:.3f}"
    )

    col4.metric(
        "F1 (Bad)",
        f"{metrics['f1']:.3f}"
    )

    col5.metric(
        "Gini",
        f"{metrics['gini']:.3f}"
    )


    st.divider()


    # =========================================================
    # 3. CONFUSION MATRIX
    # =========================================================

    st.subheader("Confusion Matrix")

    cm = artifact["confusion_matrix"]

    confusion_df = pd.DataFrame(
        cm,
        index=["Actual Good", "Actual Bad"],
        columns=["Predicted Good", "Predicted Bad"]
    )

    fig_cm = px.imshow(
        confusion_df,
        text_auto=True,
        labels={
            "x": "Predicted Class",
            "y": "Actual Class",
            "color": "Borrowers"
        }
    )

    fig_cm.update_layout(
        height=450,
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20
        )
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )


    st.caption(
        "The confusion matrix shows how many borrowers were "
        "correctly and incorrectly classified as Good or Bad."
    )


    st.divider()


    # =========================================================
    # 4. MODEL INTERPRETATION
    # =========================================================

    st.subheader("Performance Interpretation")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Bad Loan Detection")

        st.write(
            f"The model identifies **{metrics['recall']:.1%}** of actual "
            "Bad loans, meaning some Bad loans are not detected."
        )

        st.write(
            f"Among borrowers predicted as Bad, "
            f"**{metrics['precision']:.1%}** are actually Bad."
        )


    with col2:

        st.markdown("### Model Discrimination")

        st.write(
            f"The ROC-AUC of **{metrics['roc_auc']:.3f}** measures the "
            "model's ability to distinguish between Good and "
            "Bad loans across different classification thresholds."
        )

        st.write(
            f"The PR-AUC of **{metrics['pr_auc']:.3f}** focuses on the "
            "precision-recall trade-off for the relatively "
            "rare Bad-loan class."
        )


    st.divider()


    # =========================================================
    # 5. THRESHOLD INFORMATION
    # =========================================================

    st.subheader("Classification Threshold")

    st.write(
        f"The model uses a probability threshold of "
        f"**{threshold:.2f}**. Borrowers with a predicted "
        "probability of Bad loan at or above this threshold "
        "are classified as Bad."
    )

    st.info(
        "The threshold was selected using out-of-fold training "
        "predictions to maximise F1-score. The test set was "
        "kept separate for final evaluation."
    )