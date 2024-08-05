import pickle
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# initializing the commodities and encoding them to match the input required by model
all_commodities = ['Cassava', 'Maize', 'Millet']
commodity_encoder = LabelEncoder()
commodity_encoder.fit(all_commodities)

# saving the LabelEncoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(commodity_encoder, f)

# loading the saved model
model = load_model('/home/zviko/Desktop/model(4).keras')

# loading the LabelEncoder
with open('label_encoder.pkl', 'rb') as f:
    commodity_encoder = pickle.load(f)

# initializing the scaler
scaler = StandardScaler()

def encode_commodity(commodity):
    """ Encoding commodities using the LabelEncoder """
    return commodity_encoder.transform([commodity])[0]

def preprocess_input(prices, dates, commodity_encoded):
    """ Preprocessing input data """
    prices = np.array(prices).reshape(-1, 1)
    dates = np.array(dates).reshape(-1, 1)
    commodity_encoded = np.array([commodity_encoded] * len(prices)).reshape(-1, 1)
    
    # scaling inputs
    prices_scaled = scaler.fit_transform(prices).flatten()
    dates_scaled = scaler.fit_transform(dates).flatten()
    
    # combining and reshaping features to match the model's input shape
    features = np.column_stack((prices_scaled, dates_scaled, commodity_encoded)).reshape(1, -1, 3)
    
    return features

def denormalize_predictions(predictions):
    """ Denormalizing predictions and ensuring non-negative values """
    denormalized = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    # to ensure all predictions are non-negative
    return np.maximum(denormalized, 0)


def predict_multiple_days(model, initial_sequence, num_days):
    """ Predicting multiple future days """
    predictions = []
    current_sequence = np.array(initial_sequence).reshape(1, -1, 3)
    for _ in range(num_days):
        next_steps = model.predict(current_sequence)
        predictions.append(next_steps[0, -1])

        # updating the sequence to include the latest predictions
        current_sequence = np.roll(current_sequence, shift=-1, axis=1)
        current_sequence[0, -1, 0] = next_steps[0, -1]  # Update the latest price
    return denormalize_predictions(predictions)


st.title('Price Prediction for Goods')

# user-input fields
date = st.date_input("Select Date", key="select_date")
commodity = st.selectbox("Select Commodity", all_commodities, key="select_commodity")
prices_input = st.text_input("Enter Last 5 Prices (comma-separated)", "9.6,4.5,8.9,6.7,8.7", key="prices_input")
num_days = st.number_input("Select Number of Days to Predict", min_value=1, max_value=30, value=3, key="num_days")

# use a unique key for the start date input
start_date = st.date_input("Start Date", key="start_date")

if prices_input:
    prices = list(map(float, prices_input.split(',')))

    # calculating length from initial reference date to predicted date
    dates = [(date - start_date).days for _ in range(len(prices))]

    # encoding commodities
    commodity_encoded = encode_commodity(commodity)
    
    # preprocessing user-input data
    processed_input = preprocess_input(prices, dates, commodity_encoded)

    # button to activate the app's predction operation
    if st.button("Predict Price"):
        future_predictions = predict_multiple_days(model, processed_input, num_days)
        future_predictions = denormalize_predictions(future_predictions)
        
        # predict the price for the next day - one day
        prediction = model.predict(processed_input)
        prediction = denormalize_predictions([prediction[0, -1]])[0]
        
        # displaying the predicted price
        st.write(f"Predicted Price for the next day: ${prediction:.2f}")
        st.write(f"Predicted Price for the next {num_days} days: ${future_predictions}")

        # plotting the prices and future predictions
        fig, ax = plt.subplots()

        # previous prices
        historical_prices = prices

        # concatinating historical prices with predicted future prices
        all_prices = historical_prices + list(future_predictions)

        x_historical = range(1, len(historical_prices) + 1)
        x_predicted = range(len(historical_prices) + 1, len(historical_prices) + num_days + 1)

        # plotting historical prices
        ax.plot(x_historical, historical_prices, label='Historical Prices', marker='o')
        # plotting new predicted prices
        ax.plot(x_predicted, future_predictions, 'r--', label='Predicted Prices')

        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title(f"Price Changes for {commodity}")
        ax.legend()

        # displaying the plot
        st.pyplot(fig)
