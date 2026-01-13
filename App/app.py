import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Air Quality Classification System",
    page_icon="🌍",
    layout="wide"
)

# --------------------------------------------------
# Load Data & Models
# --------------------------------------------------
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    return pd.read_csv(f"{BASE_DIR}/Data/air_quality.csv")

@st.cache_resource
def load_models():
    BASE_DIR = Path(__file__).resolve().parent.parent
    return {
        "Logistic Regression": joblib.load(f"{BASE_DIR}/Models/logistic_regression.pkl"),
        "SVM": joblib.load(f"{BASE_DIR}/Models/SVM.pkl"),
        "Decision Tree": joblib.load(f"{BASE_DIR}/Models/decision_tree.pkl"),
        "Random Forest": joblib.load(f"{BASE_DIR}/Models/random_forest.pkl")
    }

df = load_data()
models = load_models()

label_map = {
    'Poor': 0,
    'Moderate': 1,
    'Good': 2,
    'Hazardous': 3
}
reverse_label_map = {
    0: "Poor",
    1: "Moderate",
    2: "Good",
    3: "Hazardous"
}


# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Overview",
        "📊 Data Exploration",
        "🤖 Model Performance",
        "🧪 Air Quality Prediction",
        "👨‍💻 About Project"
    ]
)

# ==================================================
# OVERVIEW
# ==================================================
if menu == "🏠 Overview":
    st.title("🌍 Air Quality Classification System")

    st.markdown("""
    **An end-to-end Machine Learning application that classifies air quality levels 
    using environmental and demographic indicators.**
    """)

    st.markdown("""
    ### Problem Statement
    Air pollution poses significant health risks worldwide.  
    Accurate classification of air quality levels helps governments, urban planners, 
    and environmental agencies take preventive and corrective actions.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Features Used", df.shape[1] - 1)
    col3.metric("Regions Covered", "Multiple")
    col4.metric("Air Quality Classes", df["Air Quality"].nunique())

    st.success(
        "This project demonstrates a complete ML workflow — from data analysis to a deployed predictive system."
    )

# ==================================================
# DATA EXPLORATION
# ==================================================
elif menu == "📊 Data Exploration":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Air Quality Class Distribution")
    st.bar_chart(df["Air Quality"].value_counts())

    st.info("Most samples fall under Moderate and Good categories.")

    st.divider()

    st.subheader("PM2.5 Levels Across Air Quality Categories")
    fig, ax = plt.subplots()
    sns.boxplot(x="Air Quality", y="PM2.5", data=df, ax=ax)
    st.pyplot(fig)

    st.divider()

    st.subheader("PM10 Levels Across Air Quality Categories")
    fig, ax = plt.subplots()
    sns.boxplot(x="Air Quality", y="PM10", data=df, ax=ax)
    st.pyplot(fig)

    st.divider()

    st.subheader("Correlation Between Pollutants")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        df.drop(columns=["Air Quality"]).corr(),
        annot=True,          # ✅ show values
        fmt=".2f",           # ✅ 2 decimal places
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)


    st.success(
        "EDA clearly shows that particulate matter (PM2.5, PM10) and gaseous pollutants strongly influence air quality."
    )

# ==================================================
# MODEL PERFORMANCE
# ==================================================
elif menu == "🤖 Model Performance":
    st.title("🤖 Model Evaluation & Comparison")

    st.markdown("""
    Multiple machine learning models were trained and evaluated.
    Random Forest achieved the best performance by learning non-linear
    pollution thresholds.
    """)

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "SVM", "Decision Tree", "Random Forest"],
        "Accuracy (%)": [85, 70, 91, 95]
    })

    st.table(results)

    st.divider()

    selected_model_name = st.selectbox(
        "Select Model to View Confusion Matrix",
        list(models.keys())
    )

    model = models[selected_model_name]

    X = df.drop(columns=["Air Quality"])
    y = df["Air Quality"].map(label_map)

    y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_map.keys(),
        yticklabels=label_map.keys(),
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {selected_model_name}")
    st.pyplot(fig)

    # ------------------------------
    # Classification Report
    # ------------------------------
    st.subheader("📄 Classification Report")

    report = classification_report(
        y,
        y_pred,
        target_names=label_map.keys(),
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    # Display nicely as a table
    st.dataframe(
        report_df.style.format("{:.2f}"),
        width="stretch"
    )
    # st.text("Classification Report")
    # st.text(classification_report(y, y_pred, target_names=label_map.keys()))

# ==================================================
# PREDICTION
# ==================================================
elif menu == "🧪 Air Quality Prediction":
    st.title("🧪 Real-Time Air Quality Prediction")

    selected_model_name = st.selectbox(
        "Select Machine Learning Model",
        list(models.keys())
    )
    model = models[selected_model_name]

    st.markdown("### Enter Environmental Parameters")

    col1, col2 = st.columns(2)

    with col1:
        temp = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
        pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 300.0, 50.0)
        pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 80.0)

    with col2:
        no2 = st.slider("NO2 (ppb)", 0.0, 200.0, 40.0)
        so2 = st.slider("SO2 (ppb)", 0.0, 200.0, 20.0)
        co = st.slider("CO (ppm)", 0.0, 10.0, 1.0)
        industry = st.slider("Distance to Industrial Area (km)", 0.0, 50.0, 10.0)
        population = st.number_input("Population Density (people/km²)", 100, 50000, 5000)

    input_df = pd.DataFrame([[
        temp, humidity, pm25, pm10, no2, so2, co, industry, population
    ]])

    if st.button("Predict Air Quality"):
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Air Quality Level: **{reverse_label_map[int(pred)]}**")

# ==================================================
# ABOUT
# ==================================================
elif menu == "👨‍💻 About Project":
    st.title("👨‍💻 About the Project")

    st.markdown("""
    ### 🌍 Air Quality Classification System

    This project classifies air quality into **Good, Moderate, Poor, and Hazardous**
    using environmental and demographic indicators.
    """)

    st.markdown("""
    **Key Highlights**
    - Performed detailed EDA to understand pollution patterns
    - Compared multiple ML models
    - Selected Random Forest for best performance
    - Built a fully interactive Streamlit application
    - Designed for real-world interpretability and deployment
    """)

    st.markdown("""
    **Skills Demonstrated**
    - Python & Data Analysis  
    - Machine Learning (Classification)  
    - Model Evaluation & Explainability  
    - Streamlit Deployment  
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.link_button("💼 LinkedIn", "https://www.linkedin.com/in/kornu-sai-govinda-rao-b077a9286/")
    with col2:
        st.link_button("📂 GitHub", "https://github.com/sai-govinda-rao/Air-Pollution-Analysis-Classification-")
