import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

# Synthetic Data Generation
dates = pd.date_range(start="2020-01-01", periods=365, freq='D')
soil_types = ['sandy', 'loamy', 'clayey']

data = {
    'date': dates,
    'crop_type': np.random.choice(['rice', 'wheat', 'corn'], 365),
    'soil_type': np.random.choice(soil_types, 365),
    'supply': [i + 3*np.sin(i/15) + 4*np.random.rand() for i in range(365)],
    'demand': [i + 3*np.sin(i/30) + 5*np.random.rand() for i in range(365)],
    'rainfall': np.random.rand(365) * 100,
    'temperature': np.random.rand(365) * 35,
    'humidity': np.random.rand(365) * 100
}

df = pd.DataFrame(data)
df['day_of_year'] = df['date'].dt.dayofyear
df['profit'] = df['demand'] - df['supply']
df['prev_crop'] = df['crop_type'].shift(1).fillna(method='bfill')

features = ['day_of_year', 'crop_type', 'prev_crop', 'soil_type']
X = df[features]
X = pd.get_dummies(X, columns=['crop_type', 'prev_crop', 'soil_type'], drop_first=True)

y_supply = df['supply']
y_demand = df['demand']

X_train, X_test, y_train_supply, y_test_supply = train_test_split(X, y_supply, test_size=0.2, random_state=42)
_, _, y_train_demand, y_test_demand = train_test_split(X, y_demand, test_size=0.2, random_state=42)

supply_model = LinearRegression().fit(X_train, y_train_supply)
demand_model = LinearRegression().fit(X_train, y_train_demand)

def predict_supply_demand_gap(date, prev_crop, soil_type):
    day_of_year = pd.to_datetime(date).dayofyear
    crops = ['rice', 'wheat', 'corn']
    crops_with_gap = []

    for crop in crops:
        input_data_dict = {col: 0 for col in X.columns}
        input_data_dict['day_of_year'] = day_of_year
        if f'crop_type_{crop}' in X.columns:
            input_data_dict[f'crop_type_{crop}'] = 1
        if f'prev_crop_{prev_crop}' in X.columns:
            input_data_dict[f'prev_crop_{prev_crop}'] = 1
        if f'soil_type_{soil_type}' in X.columns:
            input_data_dict[f'soil_type_{soil_type}'] = 1

        input_data = pd.DataFrame([input_data_dict])

        predicted_supply = supply_model.predict(input_data)[0]
        predicted_demand = demand_model.predict(input_data)[0]

        if predicted_demand > predicted_supply:
            crops_with_gap.append(crop)

    return crops_with_gap

crop_models = {}
for crop in ['rice', 'wheat', 'corn']:
    crop_df = df[df['crop_type'] == crop]
    if not crop_df.empty:
        model = ARIMA(crop_df['profit'], exog=crop_df[['rainfall', 'temperature', 'humidity']], order=(5,1,0))
        model_fit = model.fit()
        crop_models[crop] = model_fit

def recommend_best_crop_given_climate(date, prev_crop, predicted_rainfall, predicted_temperature, predicted_humidity, soil_type):
    crops_with_gap = predict_supply_demand_gap(date, prev_crop, soil_type)

    # Filter out the previous crop from the recommendation list
    if prev_crop in crops_with_gap:
        crops_with_gap.remove(prev_crop)

    if not crops_with_gap:
        return "No new crops with a supply-demand gap for the given date. Consider alternatives."

    best_crop = None
    best_profit = float('-inf')
    exog_data = pd.DataFrame({'rainfall': [predicted_rainfall], 'temperature': [predicted_temperature], 'humidity': [predicted_humidity]})

    for crop in crops_with_gap:
        model = crop_models.get(crop)
        if model:
            try:
                forecast_results = model.get_forecast(steps=1, exog=exog_data)
                predicted_profit = forecast_results.predicted_mean.iloc[0]
                if predicted_profit > best_profit:
                    best_profit = predicted_profit
                    best_crop = crop
            except Exception as e:
                print(f"Error forecasting for crop {crop}: {e}")

    return best_crop

st.title("Crop Recommendation System")
D = st.date_input("Date")
PC = st.selectbox('Previous Crop',('Rice', 'Wheat', 'Corn'))
T = st.number_input(label = "Temperature")
R = st.number_input(label = "Rainfall")
H = st.number_input(label = "Humidity")
S = st.selectbox('Soil Type',('sandy', 'loamy', 'clayey'))
best_crop = recommend_best_crop_given_climate(D,PC,T,R,H,S)
if st.button("enter"):
  st.write(best_crop)
