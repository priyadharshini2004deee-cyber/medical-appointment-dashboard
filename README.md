# 🏥 Medical Appointment No-Show Prediction & Demand Forecasting

## 📌 Project Overview
This project is a Machine Learning + Streamlit Dashboard designed to predict patient no-shows and forecast future appointment demand. It helps hospitals improve efficiency, reduce revenue loss, and make better decisions using data.

---

## 🎯 Problem Statement
Hospitals face several challenges:
- High no-show rate (~31.8%)
- Unpredictable patient demand
- Poor staff planning
- Resource wastage
- No early identification of risky patients

---

## 🚀 Solution
This project provides two main solutions:

### 🔹 No-Show Prediction
- Predicts whether a patient will attend or miss an appointment
- Uses patient details, health conditions, appointment info, and weather

### 🔹 Demand Forecasting
- Predicts future patient count
- Helps hospitals plan staff and resources

---

## 📊 Dashboard Features
The Streamlit dashboard includes:

- 📊 Overview Dashboard
- 👤 Patient Explorer
- 🌦️ Weather Analysis
- 🩺 Specialty Analysis
- 🗺️ Place Analysis
- 🤖 Prediction Module
- 📅 Forecast Module
- 📈 Time Series Analysis
- 🧪 Train-Test Metrics
- 🚨 Patient Alerts
- 📅 Date-wise Analysis
- 📋 Data Explorer

---

## 🧠 Machine Learning
- Random Forest Classifier (No-show prediction)
- Random Forest Regressor (Forecasting)
- SMOTE used for handling imbalanced data

---

## 📁 Project Structure

medical-appointment-dashboard/
│
├── app.py
├── Medical_appointment_data.csv
├── best_noshow_model.pkl
├── demand_forecast_model.pkl
├── noshow_feature_columns.pkl
├── forecast_feature_columns.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── .gitignore

---

## ⚙️ Installation

pip install -r requirements.txt

---

## ▶️ Run the Application

streamlit run app.py

---

## 📊 Dataset Information
- ~109,000 rows
- Target column: no_show (Yes/No)

Features include:
- Patient details (age, gender)
- Health conditions (diabetes, hypertension)
- Appointment details (date, shift, specialty)
- Weather data (temperature, rainfall)

---

## 💡 Business Impact
- Reduce no-show losses
- Improve hospital efficiency
- Better staff scheduling
- Data-driven decision making

---

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly

---

## 👩‍💻 Author
Priyadharshini