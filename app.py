"""
BSNL Optical Fibre Fault Prediction Dashboard
Run: streamlit run dashboard/app.py
"""

import os, sys, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL    = os.path.join(BASE, "models")
DATA     = os.path.join(BASE, "data", "sensor_data.csv")
FEATURES = ["opm_dbm","temperature","humidity","vibration","rainfall_mm"]

FAULT_COLORS = {
    "Normal":              "#10B981",
    "Water_Ingress":       "#EF4444",
    "Thermal_Fault":       "#F97316",
    "Physical_Cut":        "#DC2626",
    "Loose_Connector":     "#F59E0B",
    "Gradual_Degradation": "#A78BFA",
}
FAULT_SEVERITY = {
    "Normal":              "🟢 NORMAL",
    "Water_Ingress":       "🔴 URGENT",
    "Thermal_Fault":       "🟠 HIGH",
    "Physical_Cut":        "🔴 URGENT",
    "Loose_Connector":     "🟡 MEDIUM",
    "Gradual_Degradation": "🟡 MEDIUM",
}
FAULT_ACTIONS = {
    "Normal":              "System operating normally. Continue monitoring.",
    "Water_Ingress":       "🚨 Dispatch field team immediately! Check underground ducts and splice boxes for water flooding. Kerala monsoon risk is HIGH.",
    "Thermal_Fault":       "⚠️ Check OLT cabinet cooling fans and ventilation. Thermal expansion may damage fibre joints.",
    "Physical_Cut":        "🚨 Severe signal loss — likely cable cut. Inspect area for digging or accidents nearby.",
    "Loose_Connector":     "🔧 Inspect and re-seat connectors at the flagged junction. Intermittent fault pattern.",
    "Gradual_Degradation": "🔧 Schedule preventive maintenance. Signal declining slowly — plan replacement before failure.",
}
LOCATIONS = ["OLT_Main","Splitter_A","Splitter_B","Splitter_C",
             "DistBox_1","DistBox_2","DistBox_3","Junction_1","Junction_2"]

st.set_page_config(
    page_title="BSNL Fault Prediction System",
    page_icon="🔆", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background-color: #0D1B3E; }
section[data-testid="stSidebar"] { background-color: #111f45; border-right: 1px solid #2D3F6E; }
div[data-testid="metric-container"] {
    background: #1A2B52; border-radius: 12px;
    padding: 12px; border: 1px solid #2D3F6E;
}
.stButton > button {
    background: linear-gradient(135deg,#7C3AED,#5B21B6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 15px !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4);
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return (joblib.load(os.path.join(MODEL,"random_forest.pkl")),
                joblib.load(os.path.join(MODEL,"label_encoder.pkl")),
                joblib.load(os.path.join(MODEL,"scaler.pkl")))
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_data():
    return pd.read_csv(DATA, parse_dates=["timestamp"]) if os.path.exists(DATA) else None

def plotly_base():
    return dict(paper_bgcolor="#0D1B3E", plot_bgcolor="#111f45",
                font=dict(color="#FFFFFF", family="Segoe UI"),
                margin=dict(l=20,r=20,t=40,b=20))

rf, le, scaler = load_model()
df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔆 BSNL FaultAI")
    st.caption("Optical Fibre Fault Prediction | Kaithamukku Exchange")
    st.divider()
    page = st.radio("Navigation", [
        "🏠 Live Prediction",
        "📊 Analytics",
        "📋 Sensor Data",
        "ℹ️ About"
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("**System Status**")
    if rf:
        st.success("✅ ML Model Active")
        st.info(f"📦 Classes: {len(le.classes_)}")
    else:
        st.error("❌ Run train_model.py first")
    if df is not None:
        st.success(f"✅ Data: {len(df):,} records")
    st.divider()
    st.caption("Sadhika Sunil B | CSE AI & ML\nBSNL Internship 2026")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if "Live Prediction" in page:
    st.title("🔆 Live Fault Prediction")
    st.caption("Enter real-time sensor values to predict optical fibre fault type.")
    st.divider()

    left, right = st.columns([1,1], gap="large")

    with left:
        st.subheader("📡 Sensor Input Panel")
        location  = st.selectbox("📍 Network Location", LOCATIONS)
        opm       = st.slider("OPM Signal Level (dBm)",  -50.0,-10.0,-17.0, 0.1)
        temp      = st.slider("🌡️ Temperature (°C)",      10.0, 90.0, 32.0, 0.5)
        humidity  = st.slider("💧 Humidity (%)",          20.0,100.0, 58.0, 1.0)
        vibration = st.slider("📳 Vibration Level",        0.0,  1.0,  0.02,0.01)
        rainfall  = st.slider("🌧️ Rainfall (mm)",          0.0,100.0,  2.0, 0.5)

        if opm >= -20:
            st.success(f"OPM Status: 🟢 Normal ({opm} dBm)")
        elif opm >= -28:
            st.warning(f"OPM Status: 🟡 Low ({opm} dBm)")
        else:
            st.error(f"OPM Status: 🔴 Critical ({opm} dBm)")

        predict_btn = st.button("🔍 Predict Fault Now", use_container_width=True)

    with right:
        st.subheader("🤖 Prediction Result")
        if predict_btn:
            if not rf:
                st.error("Model not loaded. Run `python train_model.py` first.")
            else:
                X         = scaler.transform(pd.DataFrame([[opm,temp,humidity,vibration,rainfall]], columns=FEATURES))
                label     = rf.predict(X)[0]
                proba     = rf.predict_proba(X)[0]
                fault     = le.inverse_transform([label])[0]
                confidence= proba[label] * 100
                sev       = FAULT_SEVERITY[fault]
                action    = FAULT_ACTIONS[fault]
                color     = FAULT_COLORS[fault]

                # Result display using native Streamlit
                st.metric("Detected Fault", fault.replace("_"," "), sev)
                st.metric("Confidence", f"{confidence:.1f}%")
                st.progress(confidence / 100)

                if "URGENT" in sev:
                    st.error(f"**Severity:** {sev}\n\n**Action:** {action}")
                elif "HIGH" in sev:
                    st.warning(f"**Severity:** {sev}\n\n**Action:** {action}")
                elif "MEDIUM" in sev:
                    st.warning(f"**Severity:** {sev}\n\n**Action:** {action}")
                else:
                    st.success(f"**Severity:** {sev}\n\n**Action:** {action}")

                st.divider()
                # Probability chart
                prob_df = pd.DataFrame({
                    "Fault": [c.replace("_"," ") for c in le.classes_],
                    "Probability %": proba * 100,
                    "Color": [FAULT_COLORS[c] for c in le.classes_]
                }).sort_values("Probability %", ascending=True)

                fig = go.Figure(go.Bar(
                    x=prob_df["Probability %"], y=prob_df["Fault"],
                    orientation="h",
                    marker_color=prob_df["Color"].tolist(),
                    text=[f"{v:.1f}%" for v in prob_df["Probability %"]],
                    textposition="outside", textfont=dict(color="white", size=11)
                ))
                fig.update_layout(**plotly_base(),
                    title="Fault Probability Breakdown",
                    height=260,
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(gridcolor="#2D3F6E"))
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👈 Adjust sensor values on the left and click **Predict Fault Now**")
            st.markdown("""
            | Sensor | Normal Range |
            |--------|-------------|
            | OPM | -18 to -14 dBm |
            | Temperature | 25 – 45 °C |
            | Humidity | 40 – 65 % |
            | Vibration | 0.0 – 0.05 |
            | Rainfall | 0 – 5 mm |
            """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.title("📊 Network Analytics")
    st.caption("Historical fault patterns and sensor correlation analysis.")
    st.divider()

    if df is None:
        st.error("No data. Run `python data/generate_data.py` first.")
        st.stop()

    faults = df[df["fault_type"] != "Normal"]
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Records",  f"{len(df):,}")
    k2.metric("Fault Events",   f"{len(faults):,}", f"{len(faults)/len(df)*100:.1f}%")
    k3.metric("Urgent Faults",  f"{len(df[df['fault_type'].isin(['Water_Ingress','Physical_Cut'])]):,}")
    k4.metric("Avg OPM (dBm)", f"{df['opm_dbm'].mean():.2f}")

    st.divider()
    c1,c2 = st.columns(2)
    with c1:
        fc = df["fault_type"].value_counts().reset_index()
        fc.columns = ["Fault","Count"]
        fig = px.pie(fc, names="Fault", values="Count", hole=0.4,
                     color="Fault", color_discrete_map=FAULT_COLORS,
                     title="Fault Type Distribution")
        fig.update_layout(**plotly_base(), height=340,
                          legend=dict(bgcolor="#1A2B52",bordercolor="#2D3F6E"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(df, x="fault_type", y="opm_dbm",
                     color="fault_type", color_discrete_map=FAULT_COLORS,
                     title="OPM Signal Level by Fault Type")
        fig.update_layout(**plotly_base(), height=340,
                          showlegend=False, xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        s = df.sample(600, random_state=42)
        fig = px.scatter(s, x="humidity", y="opm_dbm",
                         color="fault_type", color_discrete_map=FAULT_COLORS,
                         opacity=0.7, title="Humidity vs OPM Signal")
        fig.update_layout(**plotly_base(), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        lf = faults["location"].value_counts().reset_index()
        lf.columns = ["Location","Count"]
        fig = px.bar(lf, x="Count", y="Location", orientation="h",
                     color="Count", color_continuous_scale=["#1A2B52","#EF4444"],
                     title="Fault Count by Network Location")
        fig.update_layout(**plotly_base(), height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔗 Sensor Correlation Matrix")
    corr = df[FEATURES].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="Feature Correlation Heatmap")
    fig.update_layout(**plotly_base(), height=380)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SENSOR DATA
# ══════════════════════════════════════════════════════════════════════════════
elif "Sensor Data" in page:
    st.title("📋 Sensor Data Log")
    st.caption("Browse and filter all historical sensor readings.")
    st.divider()

    if df is None:
        st.error("No data. Run `python data/generate_data.py` first.")
        st.stop()

    f1,f2 = st.columns(2)
    with f1:
        fault_filter = st.multiselect("Filter Fault Type",
            options=df["fault_type"].unique().tolist(),
            default=df["fault_type"].unique().tolist())
    with f2:
        loc_filter = st.multiselect("Filter Location",
            options=df["location"].unique().tolist(),
            default=df["location"].unique().tolist())

    filtered = df[df["fault_type"].isin(fault_filter) & df["location"].isin(loc_filter)]
    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.dataframe(
        filtered[["timestamp","location","opm_dbm","temperature",
                  "humidity","vibration","rainfall_mm","fault_type"]].reset_index(drop=True),
        use_container_width=True, height=320
    )

    fig = px.line(
        filtered.sort_values("timestamp").head(400),
        x="timestamp", y="opm_dbm", color="fault_type",
        color_discrete_map=FAULT_COLORS,
        title="OPM Signal Over Time"
    )
    fig.update_layout(**plotly_base(), height=300)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif "About" in page:
    st.title("ℹ️ About This Project")
    st.divider()

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("🎯 Problem Statement")
        st.info("""
        Optical fibre faults at BSNL Kaithamukku are identified only through
        **customer complaints**, causing long resolution times.

        This AI system enables **proactive fault detection** before customers are affected.
        """)

        st.subheader("🤖 ML Pipeline")
        st.success("""
        📥 Multi-sensor data (5 features)
        🔧 Feature engineering + Standard Scaling
        🌳 Random Forest (200 estimators)
        ⚡ XGBoost (comparison model)
        📊 5-fold Cross Validation
        🎯 ~100% accuracy on simulated data
        """)

    with c2:
        st.subheader("🔬 Sensor Stack")
        st.markdown("""
        | Sensor | Detects |
        |--------|---------|
        | OPM | Signal loss & attenuation |
        | Temperature | Thermal/equipment faults |
        | Moisture | Water ingress (Kerala monsoon) |
        | Vibration | Physical cuts & damage |
        | Weather API | Environment correlation |
        """)

        st.subheader("🛠️ Tech Stack")
        st.markdown("`Python` · `Scikit-learn` · `XGBoost` · `Streamlit` · `Plotly` · `Pandas` · `NumPy`")

        st.subheader("👩‍💻 Developed By")
        st.markdown("""
        **Sadhika Sunil B** | Dept. CSE (AI & ML)
        BSNL Kaithamukku Exchange | June 2026
        🤝 Hardware: EC Intern | Software: CSE AI/ML Intern
        """)
