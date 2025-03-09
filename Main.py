import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# Ensure scikit-learn compatibility

# Load the trained model safely
@st.cache_resource
def load_model():
    try:
        model = joblib.load('house_price_model.pkl') 
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model()

# Streamlit App UI
st.title("🏡 House Price Prediction App")
st.write("Enter house details to predict the price.")

# Sidebar for user input
st.sidebar.header("🔹 Input House Features")

# Define user input fields
Avg._Area_Income = st.sidebar.number_input("Avg. Area Income", min_value=500, max_value=100000, step=100)
Avg._Area_House_Age = st.sidebar.selectbox("Avg. Area House Age", options=list(range(1, 11)))

# Ensure model is loaded
if model:
    # Prepare input features (Ensure column names match training data)
    feature_names = ['Avg. Area Income', 'Avg. Area House Age']  # Adjust based on training data
    features = pd.DataFrame([[lot_area, overall_quality]], columns=feature_names)

    # Predict when the user presses the button
    if st.sidebar.button("🔍 Price"):
        try:
            prediction = model.predict(features)[0]
            st.sidebar.success(f"🏠 **Estimated Price:** ${prediction:,.2f}")
        except Exception as e:
            st.sidebar.error(f"⚠️ Prediction Error: {e}")

# Display dataset information
st.subheader("📊 Dataset Overview")
try:
    train = pd.read_csv('/kaggle/input/help-predicting-housing-prices-usa/USA_Housing.csv')  # Ensure correct path to dataset

    st.write(train.head())

    # Plot SalePrice distribution
    st.subheader("📉 Price Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(train['Price'], kde=True, bins=30, ax=ax)
    ax.set_title("Distribution of Price")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

except Exception as e:
    st.warning(f"⚠️ Could not load dataset: {e}")
