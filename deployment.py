import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Title and description
st.title("Energy Production Prediction App")
st.write("""
This app predicts **Energy Production** based on various cooling parameters.
Enter the values in the sidebar and see the prediction update in real-time!
""")

# Sidebar input for user parameters
st.sidebar.header('User Input Parameters')

def user_input_features():
    amb_pressure = st.sidebar.number_input('Ambient Pressure (atm)')
    temperature = st.sidebar.number_input('Temperature (°C)')
    exhaust_vacuum = st.sidebar.number_input('Exhaust Vacuum (cm Hg)')
    r_humidity = st.sidebar.number_input('Relative Humidity (%)')

    data = {
        'amb_pressure': amb_pressure,
        'temperature': temperature,
        'exhaust_vacuum': exhaust_vacuum,
        'r_humidity': r_humidity
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Load data and model
data = pd.read_csv(r"C:\Users\ompre\OneDrive\Desktop\ExcelR\ExcelR Internship Projects\Internship Project 1\Copy of Regrerssion_energy_production_data (2).csv", delimiter=';')
X = data[['amb_pressure', 'temperature', 'exhaust_vacuum', 'r_humidity']]
y = data['energy_production']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, max_features=0.7, max_samples=0.6)
model.fit(X_train_scaled, y_train)

# Collect and display user input
user_input = user_input_features()
st.write("### User Input Parameters:")
st.write(user_input.to_html(index=False), unsafe_allow_html=True)

# Ensure user input matches the feature names used during training
user_input_scaled = scaler.transform(user_input)

# Predict based on user input
prediction = model.predict(user_input_scaled)

# Display prediction
st.write("### Energy Production Prediction:")
st.write(f"The predicted energy production is: **{prediction[0]:.2f} MW**")

# Display model performance
y_pred_test = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)
r2 = r2_score(y_test, y_pred_test)

st.write("### Model Performance on Test Data:")
st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
st.metric("R-squared (R2)", f"{r2:.2f}")
