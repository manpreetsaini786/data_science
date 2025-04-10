import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set Streamlit Page Config
st.set_page_config(page_title="Anxiety Attack Analysis", page_icon="😰", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Click Below")
page = st.sidebar.radio("Go to", ["Home", "Data Visualization", "Predictions", "Reports"])

# 📌 Sidebar Section - File Upload
st.sidebar.title("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Initialize DataFrame as None
df = None

# Load Data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove extra spaces from column names

    # Ensure required columns exist
    required_columns = ["Gender", "Occupation", "Stress Level (1-10)", "Heart Rate (bpm during attack)",
                        "Breathing Rate (breaths/min)", "Caffeine Intake (mg/day)", "Alcohol Consumption (drinks/week)",
                        "Severity of Anxiety Attack (1-10)", "Sleep Hours"]

    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.sidebar.error(f"❌ Missing columns in dataset: {missing_cols}")
        df = None  # Prevent further execution if columns are missing
    else:
        # Convert numeric columns
        for col in required_columns[2:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=required_columns, inplace=True)

# 📌 Main Page Content
if page == "Home":
    st.title("🏠 Anxiety Attack Data Analysis")
    st.write("Analyze and predict anxiety attack severity using dataset insights.")

    if df is not None:
        st.subheader("📊 Overview Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="📉 Avg. Stress Level", value=round(df["Stress Level (1-10)"].mean(), 2))

        with col2:
            st.metric(label="❤️ Avg. Heart Rate (bpm)", value=round(df["Heart Rate (bpm during attack)"].mean(), 2))

        with col3:
            st.metric(label="🌬 Avg. Breathing Rate", value=round(df["Breathing Rate (breaths/min)"].mean(), 2))

        with col4:
            st.metric(label="☕ Avg. Caffeine Intake (mg)", value=round(df["Caffeine Intake (mg/day)"].mean(), 2))

        st.metric(label="🍷 Avg. Alcohol Intake (drinks/week)", value=round(df["Alcohol Consumption (drinks/week)"].mean(), 2))
    else:
        st.warning("⚠ Please upload a valid dataset to see insights.")

elif page == "Data Visualization":
    st.title("📊 Data Visualization")

    if df is not None:
        st.subheader("🔹 Gender vs. Anxiety Severity (Bar Chart)")
        fig, ax = plt.subplots()
        sns.barplot(x="Gender", y="Severity of Anxiety Attack (1-10)", data=df, ax=ax)
        st.pyplot(fig)

        st.subheader("🔹 Sleep Hours vs. Stress Level (Scatter Plot)")
        fig, ax = plt.subplots()
        sns.scatterplot(x="Sleep Hours", y="Stress Level (1-10)", data=df, ax=ax)
        st.pyplot(fig)

        st.subheader("🔹 Heart Rate & Breathing Rate Distribution (Histogram)")
        fig, ax = plt.subplots()
        sns.histplot(df["Heart Rate (bpm during attack)"], kde=True, color="blue", label="Heart Rate")
        sns.histplot(df["Breathing Rate (breaths/min)"], kde=True, color="red", label="Breathing Rate")
        plt.legend()
        st.pyplot(fig)

        st.subheader("🔹 Heatmap (Feature Correlation)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠ Please upload a valid dataset to view visualizations.")

elif page == "Predictions":
    st.title("🤖 Anxiety Severity Prediction")

    if df is not None:
        st.write("📥 Enter new case details to predict anxiety severity.")

        # User Input Form
        gender = st.selectbox("Select Gender", df["Gender"].unique())
        occupation = st.selectbox("Select Occupation", df["Occupation"].unique())
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=50, max_value=200, value=80)
        breathing_rate = st.number_input("Breathing Rate (breaths/min)", min_value=10, max_value=40, value=20)
        caffeine = st.number_input("Caffeine Intake (mg/day)", min_value=0, max_value=500, value=100)
        alcohol = st.number_input("Alcohol Consumption (drinks/week)", min_value=0, max_value=10, value=2)
        sleep_hours = st.slider("Sleep Hours", 0, 12, 7)

        # Dummy Prediction Model (Replace with ML model)
        predicted_severity = np.mean([stress_level, heart_rate / 20, breathing_rate / 5, caffeine / 100, alcohol])
        st.metric(label="🧠 Predicted Anxiety Severity (1-10)", value=round(predicted_severity, 2))
    else:
        st.warning("⚠ Please upload a valid dataset to use prediction.")

elif page == "Reports":
    st.title("📄 Reports & Insights")

    if df is not None:
        # Search & Sort Table
        search_text = st.text_input("🔎 Search by Gender or Occupation")
        filtered_df = df[df["Gender"].str.contains(search_text, case=False, na=False) | df["Occupation"].str.contains(search_text, case=False, na=False)]

        st.write("📋 Filtered Dataset Preview:")
        st.dataframe(filtered_df)

        # Download Reports
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇ Download CSV Report", data=csv, file_name="Anxiety_Report.csv", mime="text/csv")
    else:
        st.warning("⚠ Please upload a dataset to generate reports.")
