from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import joblib
import numpy as np

# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------
app = FastAPI()

# Allow frontend (Streamlit) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Load ML Model (.pkl)
# ---------------------------------------------------------
model = joblib.load("model.pkl")    # Make sure file is inside backend folder!

# ---------------------------------------------------------
# MongoDB (Local)
# ---------------------------------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["customer_churn"]
collection = db["user_inputs"]

# ---------------------------------------------------------
# Request Body Schema (Inputs from UI)
# ---------------------------------------------------------
class CustomerData(BaseModel):
    age: int
    subscription_type: str
    gender: str
    contract_length: str
    usage_frequency: int
    total_spend: float
    support_calls: int


# ---------------------------------------------------------
# PREDICTION API
# ---------------------------------------------------------
@app.post("/predict")
def predict(data: CustomerData):

    # ---------- 1️⃣ Save Input to MongoDB ----------
    collection.insert_one(data.dict())

    # ---------- 2️⃣ Manual Encoding (match your training pipeline!) ----------
    subscription_map = {"Basic": 0, "Standard": 1, "Premium": 2}
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    contract_map = {"Monthly": 0, "Quarterly": 1, "Yearly": 2}

    # X input same order as training
    X = np.array([[
        data.age,
        subscription_map[data.subscription_type],
        gender_map[data.gender],
        contract_map[data.contract_length],
        data.usage_frequency,
        data.total_spend,
        data.support_calls
    ]])

    # ---------- 3️⃣ Predict Tenure ----------
    prediction = model.predict(X)[0]

    # ---------- 4️⃣ Return Result ----------
    return {
        "status": "success",
        "predicted_tenure": float(prediction)
    }


# ---------------------------------------------------------
# ROOT ENDPOINT (optional)
# ---------------------------------------------------------
@app.get("/")
def home():
    return {"message": "FastAPI backend running successfully."}
