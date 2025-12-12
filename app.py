import streamlit as st
from pymongo import MongoClient
from datetime import datetime

# ----------------------------------------------------
# MongoDB Connection (Local)
# ----------------------------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["customer_churn"]
collection = db["user_inputs"]

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------
st.set_page_config(page_title="Customer Tenure Prediction", layout="centered")

st.title("📊 Customer Tenure Prediction")
st.write("Enter customer details to predict expected tenure (months).")


# --------------- Input Fields -----------------

age = st.number_input("Age", min_value=10, max_value=100, value=30)

subscription_type = st.selectbox(
    "Subscription Type",
    ["Basic", "Standard", "Premium"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

contract_length = st.selectbox(
    "Contract Length",
    ["Monthly", "Quarterly", "Yearly"]
)

usage_frequency = st.number_input(
    "Usage Frequency",
    min_value=0,
    max_value=500,
    value=50
)

total_spend = st.number_input(
    "Total Spend (₹)",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

support_calls = st.number_input(
    "Support Calls",
    min_value=0,
    max_value=500,
    value=50
)

# ----------------------------------------------------
# Save to MongoDB
# ----------------------------------------------------
if st.button("Save Details"):
    
    record = {
        "age": age,
        "subscription_type": subscription_type,
        "gender": gender,
        "contract_length": contract_length,
        "usage_frequency": usage_frequency,
        "total_spend": total_spend,
        "support_calls": support_calls,
        "timestamp": datetime.now()
    }

    try:
        collection.insert_one(record)
        st.success("✅ User data saved successfully to MongoDB!")
        st.json(record)
    except Exception as e:
        st.error(f"❌ Error saving to database: {e}")


# ----------------------------------------------------
# Output Placeholder (Model integration later)
# ----------------------------------------------------
st.subheader("🔹 Predicted Tenure")
st.info("Prediction will appear here once ML model is integrated.")
