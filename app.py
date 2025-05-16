import streamlit as st
import pandas as pd
import pickle

st.set_page_config(layout="wide", page_title="Customer Churn Predictor", page_icon="📊")
st.title("📊 Telecom Churn Prediction System")

# Load model and encoders
model = pickle.load(open('model/churn_model.pkl', 'rb'))
label_encoders = pickle.load(open('model/label_encoders.pkl', 'rb'))
feature_columns = pickle.load(open('model/feature_columns.pkl', 'rb'))

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'input'

# -------------------- INPUT PAGE --------------------
if st.session_state.page == 'input':
    st.subheader("Enter Customer Information")
    customer_name = st.text_input("Customer Name")

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges", min_value=10.0, max_value=200.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=10.0, max_value=10000.0, value=1500.0)
    with col2:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        senior_citizen = st.selectbox("Senior Citizen", ['Yes', 'No'])
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    is_new = st.radio("Is this a new customer?", ['Yes', 'No'])

    st.markdown("""
        <style>
        .main { background-color: #fce4ec; }
        .stButton>button {
            background-color: #ec407a; color:white;
            font-size:20px; border-radius:10px;
            padding: 10px 24px;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button("🔍 Predict"):
        input_dict = {col: 0 for col in feature_columns}
        input_dict['tenure'] = tenure
        input_dict['MonthlyCharges'] = monthly_charges
        input_dict['TotalCharges'] = total_charges
        input_dict['gender'] = gender
        input_dict['SeniorCitizen'] = 1 if senior_citizen == 'Yes' else 0
        input_dict['Contract'] = contract
        input_dict['is_new_customer'] = 1 if is_new == 'Yes' else 0

        for col, le in label_encoders.items():
            if col in input_dict and not isinstance(input_dict[col], (int, float)):
                input_dict[col] = le.transform([input_dict[col]])[0] if input_dict[col] in le.classes_ else 0

        input_df = pd.DataFrame([input_dict], columns=feature_columns)
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.session_state.prediction = prediction
        st.session_state.probability = prob
        st.session_state.customer_name = customer_name
        st.session_state.page = 'result'

        st.rerun()

# -------------------- RESULT PAGE --------------------
if st.session_state.page == 'result':
    from pages.Churn_Result import show_result
    show_result(
        st.session_state.prediction,
        st.session_state.probability,
        st.session_state.customer_name
    )

    if st.button("🔄 Back to Input Page"):
        st.session_state.page = 'input'
        st.rerun()







