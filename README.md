# 🔭 BSNL AI-Based Fault Prediction System
### OPM-Integrated Optical Line Terminal (OLT) Fault Detection

An AI/ML-powered fault prediction and monitoring system for BSNL's 
optical fibre network. Integrates Optical Power Monitor (OPM) sensor 
data from OLTs to detect and classify network faults in real time — 
before they impact customers.

## 🚨 Fault Types Detected
- 🟡 Fibre Degradation — gradual power decline
- 🟡 Signal Attenuation — sudden step-drop in Rx power
- 🟡 Loose Connector — oscillating power pattern
- 🔴 Fibre Cut — catastrophic signal loss

## 🧠 ML Models
- Random Forest Classifier
- XGBoost Classifier
- StandardScaler + LabelEncoder preprocessing

## 📊 Features
- Real-time Streamlit dashboard
- OPM sensor data simulation
- Fault classification with confidence scores
- Proactive maintenance alerts before customer complaints
- Self-healing network rerouting logic

## 🛠️ Tech Stack
- Python 3.12
- scikit-learn, XGBoost, joblib
- Streamlit, pandas, numpy
- ngrok (for remote access)

## 🚀 Run Locally
git clone https://github.com/yourusername/bsnl-fault-prediction
cd bsnl_project
pip install -r requirements.txt
python train_model.py
streamlit run dashboard/app.py

## 👩‍💻 Author
Sadhika Sunil B
Department of Computer Science (AI & ML)
