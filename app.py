import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load real-world dataset (Replace with actual dataset path)
df = pd.read_csv("AI_in_HealthCare_Dataset.csv")

# Ensure column names are correctly formatted
df.columns = df.columns.str.strip()

# Check if required columns exist
required_columns = ['Technological Penetration', 'Market Influence', 'Regulatory Adaptation', 'Clinical Outcomes', 'Competitive Market Dynamics', 'Year']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns in dataset: {missing_columns}")
else:
    # Normalize function
    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

    # Calculate MDI from dataset
    def calculate_mdi(data):
        tp_norm = normalize(data['Technological Penetration'].mean(), 0, 100)
        mi_norm = normalize(data['Market Influence'].mean(), 0, 100)
        ra_norm = normalize(data['Regulatory Adaptation'].mean(), 0, 100)
        co_norm = normalize(data['Clinical Outcomes'].mean(), 0, 100)
        cmd_norm = normalize(data['Competitive Market Dynamics'].mean(), 0, 100)
        
        mdi = (0.30 * tp_norm) + (0.20 * mi_norm) + (0.15 * ra_norm) + (0.20 * co_norm) + (0.15 * cmd_norm)
        return round(mdi * 100, 2)

    # Streamlit UI
    def main():
        st.title("Market Disruption Index (MDI) for AI in Healthcare")
        
        # Display dataset
        st.subheader("Real-World AI in Healthcare Dataset")
        st.dataframe(df.head())
        
        # Calculate MDI Score
        mdi_score = calculate_mdi(df)
        st.subheader(f"Market Disruption Index Score: {mdi_score}/100")
        
        # Visualization - Line Chart for AI adoption trend
        st.subheader("AI Adoption Over Time")
        fig, ax = plt.subplots()
        df.groupby('Year')['Technological Penetration'].mean().plot(kind='line', marker='o', ax=ax)
        ax.set_ylabel("AI Adoption Rate")
        st.pyplot(fig)
        
        # Visualization - Radar Chart
        labels = ['Technological Penetration', 'Market Influence', 'Regulatory Adaptation', 'Clinical Outcomes', 'Competitive Market Dynamics']
        values = [df['Technological Penetration'].mean(), df['Market Influence'].mean(), df['Regulatory Adaptation'].mean(), df['Clinical Outcomes'].mean(), df['Competitive Market Dynamics'].mean()]
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
