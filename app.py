import streamlit as st
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -------------------------------
# App Title & Description
# -------------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="centered")

st.title("🏦 Smart Loan Approval System")
st.write(
    "This system uses **Support Vector Machines (SVM)** to predict whether a loan "
    "application is likely to be **approved or rejected** based on applicant details."
)

st.divider()

# -------------------------------
# Sidebar – Input Section
# -------------------------------
st.sidebar.header("📋 Applicant Details")

income = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=2000)

credit_history = st.sidebar.selectbox(
    "Credit History", ["Yes", "No"]
)

employment_status = st.sidebar.selectbox(
    "Employment Status",
    ["Employed", "Self-Employed", "Unemployed"]
)

property_area = st.sidebar.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# -------------------------------
# Encode Inputs
# -------------------------------
credit_history_val = 1 if credit_history == "Yes" else 0

employment_map = {
    "Employed": 2,
    "Self-Employed": 1,
    "Unemployed": 0
}
employment_val = employment_map[employment_status]

property_map = {
    "Urban": 2,
    "Semiurban": 1,
    "Rural": 0
}
property_val = property_map[property_area]

X_user = np.array([
    income,
    loan_amount,
    credit_history_val,
    employment_val,
    property_val
]).reshape(1, -1)

# -------------------------------
# Model Selection
# -------------------------------
st.subheader("🔍 Select SVM Kernel")

kernel_choice = st.radio(
    "Choose SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

# -------------------------------
# Dummy Training Data
# (For academic demonstration)
# -------------------------------
X_train = np.array([
    [3000, 1500, 1, 2, 2],
    [2000, 1800, 0, 0, 0],
    [5000, 2000, 1, 2, 1],
    [1500, 3000, 0, 1, 0],
    [6000, 2500, 1, 2, 2],
    [2500, 2800, 0, 1, 1],
])

y_train = np.array([1, 0, 1, 0, 1, 0])  # 1 = Approved, 0 = Rejected

# -------------------------------
# Build Model Based on Selection
# -------------------------------
if kernel_choice == "Linear SVM":
    model = SVC(kernel="linear", probability=True)
elif kernel_choice == "Polynomial SVM":
    model = SVC(kernel="poly", degree=3, probability=True)
else:
    model = SVC(kernel="rbf", probability=True)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", model)
])

pipeline.fit(X_train, y_train)

# -------------------------------
# Prediction Button
# -------------------------------
st.divider()

if st.button("✅ Check Loan Eligibility"):
    prediction = pipeline.predict(X_user)[0]
    confidence = pipeline.predict_proba(X_user)[0].max()

    # -------------------------------
    # Output Section
    # -------------------------------
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Kernel Used:** {kernel_choice}")
    st.write(f"**Model Confidence:** {confidence * 100:.2f}%")

    # -------------------------------
    # Business Explanation
    # -------------------------------
    st.subheader("📊 Business Explanation")

    if prediction == 1:
        st.info(
            "Based on the applicant’s **income stability**, **positive credit history**, "
            "and overall financial pattern, the model predicts that the applicant "
            "is **likely to repay the loan**."
        )
    else:
        st.warning(
            "Based on the applicant’s **credit history and income pattern**, "
            "the model predicts a **higher risk of loan default**, "
            "so the loan is unlikely to be approved."
        )
