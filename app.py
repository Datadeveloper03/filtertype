import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load trained model and preprocessing tools safely
model_path = r"C:\Users\Lenovo\Downloads\hacksindia\models\filter_recommendation_model.pkl"
scaler_path = r"C:\Users\Lenovo\Downloads\hacksindia\models\scaler.pkl"
encoder_path = r"C:\Users\Lenovo\Downloads\hacksindia\models\label_encoder.pkl"

# Try loading the model and scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
except FileNotFoundError:
    st.error("🔴 Model files not found! Ensure they are in the correct path before running the app.")
    st.stop()

# Title
st.title("⛏️ Mining Emissions Filter Recommendation")

# Sidebar Inputs
st.sidebar.header("🌱 Pollution & Water Quality Levels")
co2 = st.sidebar.slider("CO2 Level", 0.0, 2.0, 0.5)
so2 = st.sidebar.slider("SO2 Level", 0.0, 2.0, 0.1)
nox = st.sidebar.slider("NOx Level", 0.0, 2.0, 0.2)
salinity = st.sidebar.slider("Salinity Level", 0.0, 5.0, 0.0)
hardness = st.sidebar.slider("Water Hardness", 0, 300, 120)
lead = st.sidebar.slider("Lead (ppm)", 0.0, 0.1, 0.02)
mercury = st.sidebar.slider("Mercury (ppm)", 0.0, 0.1, 0.005)
arsenic = st.sidebar.slider("Arsenic (ppm)", 0.0, 0.1, 0.01)

# Function to process input data
def predict_filter(input_data):
    try:
        input_scaled = scaler.transform(input_data)
        predictions = model.predict(input_scaled)
        return label_encoder.inverse_transform(predictions)
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        return None  # Return None if an error occurs

# File Upload Section
st.sidebar.subheader("📂 Upload CSV for Bulk Prediction")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### 📊 Uploaded Data Preview:")
    st.dataframe(data.head())

    # Ensure the correct columns exist
    required_cols = ['CO2', 'SO2', 'NOx', 'Salinity', 'Hardness', 'Lead', 'Mercury', 'Arsenic']
    if all(col in data.columns for col in required_cols):
        predictions = predict_filter(data[required_cols])

        # ✅ FIX: Check if predictions is not None and has elements
        if predictions is not None and len(predictions) > 0:
            data["Filter_Type"] = predictions

            # Show results
            st.write("### ✅ Prediction Results:")
            st.dataframe(data)

            # Pie chart for filter distribution
            pie_chart = px.pie(
                names=data["Filter_Type"].value_counts().index,
                values=data["Filter_Type"].value_counts().values,
                title="Filter Type Distribution",
            )
            st.plotly_chart(pie_chart)

            # Boxplot visualization for pollution levels
            st.write("### 📊 Contaminant Levels Across Samples")
            fig = px.box(data[required_cols], title="Contaminant Distribution")
            st.plotly_chart(fig)
        else:
            st.error("⚠️ No predictions generated. Check the model or input data.")
    else:
        st.error(f"⚠️ CSV must contain columns: {required_cols}")

# Manual Input Prediction
if st.sidebar.button("🔍 Recommend Filter"):
    input_data = pd.DataFrame({
        'CO2': [co2],
        'SO2': [so2],
        'NOx': [nox],
        'Salinity': [salinity],
        'Hardness': [hardness],
        'Lead': [lead],
        'Mercury': [mercury],
        'Arsenic': [arsenic]
    })
    predicted_filter = predict_filter(input_data)

    if predicted_filter is not None and len(predicted_filter) > 0:
        # Display results
        st.subheader("🎯 Recommended Filter")
        st.write(f"**🛠 Filter Type:** {predicted_filter[0]}")

        # **Generate a Bar Chart**
        fig = px.bar(
            x=["CO2", "SO2", "NOx", "Salinity", "Hardness", "Lead", "Mercury", "Arsenic"],
            y=[co2, so2, nox, salinity, hardness, lead, mercury, arsenic],
            labels={"x": "Pollutants", "y": "Concentration"},
            title="🌍 Pollution & Water Contaminant Levels",
        )
        st.plotly_chart(fig)

        # **Explanation of Contaminants & Recommended Filters**
        st.subheader("🔬 Understanding Contaminants & Filter Solutions")
        contaminant_info = {
            "CO2": "High CO2 levels can lead to acidification. Carbon filters help in reducing CO2 concentration.",
            "SO2": "Sulfur dioxide leads to acidic water. Reverse osmosis (RO) filters are recommended.",
            "NOx": "Nitrogen oxides contribute to smog. Activated carbon filters help reduce NOx.",
            "Salinity": "High salinity affects taste and corrosion. Reverse osmosis (RO) is the best solution.",
            "Hardness": "Hard water causes scaling. Water softeners or ion-exchange filters are effective.",
            "Lead": "Lead contamination is dangerous. Activated carbon or RO filters are needed.",
            "Mercury": "Mercury affects health significantly. Advanced carbon block filters remove mercury.",
            "Arsenic": "Arsenic is highly toxic. Specialized arsenic filters or RO membranes are required."
        }

        # Displaying recommendations
        for pollutant, solution in contaminant_info.items():
            st.write(f"**{pollutant}:** {solution}")
    else:
        st.error("⚠️ Prediction failed. Check the input values and try again.")