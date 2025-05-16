import streamlit as st
import matplotlib.pyplot as plt

def show_result(prediction, probability, customer_name=None):
    st.markdown("<h2 style='color:#ec407a'>🔍 Prediction Result</h2>", unsafe_allow_html=True)

    if customer_name:
        st.markdown(f"### 👤 Customer Name: **{customer_name}**")

    if prediction == 1:
        st.markdown("### ❌ This customer is likely to churn.")
        st.error(f"⚠️ Probability of churn: {probability:.2f}")
        st.image("assets/warning.gif", width=300)  # Resize image here
    else:
        st.markdown("### ✅ This customer is not likely to churn.")
        st.success(f"✅ Probability of no churn: {(1 - probability):.2f}")
        st.image("assets/happy_customer.gif", width=250)  # Resize image here

    st.progress(int(probability * 100))

    fig,ax = plt.subplots(figsize=(3, 2))
    ax.bar(['Churn', 'No Churn'], [probability, 1 - probability], color=['#e57373', '#81c784'])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # Save to history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    st.session_state['history'].append({
        'prediction': prediction,
        'probability': probability,
        'customer_name': customer_name
    })

    






