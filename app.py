import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load real-world dataset (Replace with actual dataset path)
df = pd.read_csv("ai_healthcare_dataset.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Check if required columns exist
required_columns = ['Patient_ID', 'Age', 'Gender', 'Blood_Pressure', 'Heart_Rate', 'Temperature', 'Diagnosis', 'Medication', 'Treatment_Duration', 'Insurance_Type']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns in dataset: {missing_columns}")
else:
    # Normalize function
    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

    # Calculate MDI from dataset
    def calculate_mdi(data):
        bp_norm = normalize(data.get('blood_pressure', pd.Series([0])).mean(), 60, 180)
        hr_norm = normalize(data.get('heart_rate', pd.Series([0])).mean(), 50, 150)
        temp_norm = normalize(data.get('temperature', pd.Series([0])).mean(), 95, 105)
        td_norm = normalize(data.get('treatment_duration', pd.Series([0])).mean(), 1, 365)
        
        mdi = (0.25 * bp_norm) + (0.25 * hr_norm) + (0.25 * temp_norm) + (0.25 * td_norm)
        return round(mdi * 100, 2)

    # Streamlit UI
    def main():
        st.title("Market Disruption Index (MDI) for AI in Healthcare")
        
        # Display dataset
        st.subheader("Healthcare Dataset Overview")
        st.dataframe(df.head())
        
        # Calculate MDI Score
        mdi_score = calculate_mdi(df)
        st.subheader(f"Market Disruption Index Score: {mdi_score}/100")
        
        # Visualization - Line Chart for Heart Rate Trend
        st.subheader("Heart Rate Trends Over Patients")
        fig, ax = plt.subplots()
        df['heart_rate'].plot(kind='line', marker='o', ax=ax)
        ax.set_ylabel("Heart Rate (bpm)")
        st.pyplot(fig)
        
        # Visualization - Radar Chart
        labels = ['Blood Pressure', 'Heart Rate', 'Temperature', 'Treatment Duration']
        values = [df.get('blood_pressure', pd.Series([0])).mean(), df.get('heart_rate', pd.Series([0])).mean(), df.get('temperature', pd.Series([0])).mean(), df.get('treatment_duration', pd.Series([0])).mean()]
        values = [normalize(v, 0, 100) for v in values]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        
        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='blue', alpha=0.25)
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        
        st.pyplot(fig)
        
    if __name__ == "__main__":
        main()
