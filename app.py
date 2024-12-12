import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Title of the app
st.title('IBM Stock Price Prediction')

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('IBM_2006-01-01_to_2018-01-01.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

data = load_data()

# Display the dataset
st.write("### Dataset Overview")
st.write(data.head())

# Prepare the data
close_data = data[['Close']]
train_data = close_data[close_data.index < '2017-01-01']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)

# Function to create dataset
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(scaled_train_data, time_step=60)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Building the LSTM model
model = Sequential()
model.add(LSTM(25, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # Reduced units
model.add(Dropout(0.2))
model.add(LSTM(25, return_sequences=False))  # Reduced units
model.add(Dropout(0.2))
model.add(Dense(1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train Model button
if st.button("Train Model"):
    with st.spinner("Training the model..."):
        try:
            model.fit(X_train, y_train, epochs=20, batch_size=32)  # Reduced epochs
            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"Error during training: {e}")

# Predict and plot results
if st.button("Predict"):
    # Prepare test data
    test_data = close_data[close_data.index >= '2017-01-01']
    scaled_test_data = scaler.transform(test_data)
    
    X_test, y_test = create_dataset(scaled_test_data, time_step=60)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Predictions
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plotting
    plt.figure(figsize=(14, 5))
    
    # Adjusting the index to match the predicted prices length
    plt.plot(test_data.index[60:], test_data['Close'][60:], color='blue', label='Actual Prices')  # Starting from index 60
    plt.plot(test_data.index[60:len(predicted_prices)+60], predicted_prices, color='red', label='Predicted Prices')  # Match lengths
    plt.title('IBM Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)