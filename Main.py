import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("USA_Housing.csv")

# Ensure that only numeric columns are used for correlation calculation
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Handle missing values by filling with the mean (or you can drop if needed)
df_numeric = df_numeric.fillna(df_numeric.mean())

# Streamlit App
st.title("USA Housing Data Analysis")
st.write("This dashboard provides insights into the USA Housing dataset.")

# Display dataset overview
st.subheader("Dataset Overview")
st.write(df.head())
st.write("Shape of dataset:", df.shape)

# Summary statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Data visualization
st.subheader("Data Visualizations")

# Price Distribution
st.write("### Price Distribution")
fig, ax = plt.subplots()
sns.histplot(df['Price'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Interactive Correlation Heatmap: Let users select columns
st.write("### Interactive Correlation Heatmap")
selected_columns = st.multiselect("Select columns for correlation heatmap", df_numeric.columns.tolist(), default=df_numeric.columns.tolist())
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_numeric[selected_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Simple Linear Regression Model
st.subheader("Predicting House Prices")
features = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']

# Interactive feature selection for prediction
selected_features = st.multiselect("Select features for prediction", features, default=features)

X = df[selected_features]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display model performance
st.write("### Model Performance")
st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# User input for prediction: Interactive sliders for each feature
st.subheader("Make a Prediction")
user_input = {}
for feature in selected_features:
    user_input[feature] = st.slider(feature, min_value=float(df[feature].min()), max_value=float(df[feature].max()), value=float(df[feature].mean()), step=0.1)

user_df = pd.DataFrame([user_input])
predicted_price = model.predict(user_df)[0]
st.write(f"### Predicted Price: ${predicted_price:,.2f}")
