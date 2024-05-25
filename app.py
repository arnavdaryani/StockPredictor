import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2021-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

#Describing Data
st.subheader('Data from 2010 - 2021')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


# Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#Load my model
model = load_model('keras_model.h5')

#Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#Final Graph

st.subheader('Predictions vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


def generate_predictions(input_data, model, scaler, num_predictions):
    n_steps=100
    x_input = input_data[(len(input_data) - 100):].reshape(1,-1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    i = 0
    while i < num_predictions:
        if len(temp_input) > n_steps:
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    return lst_output


# Streamlit app
# Streamlit app
def main():
    st.title("Stock Price Prediction")
    num_predictions_input = st.text_input('Enter the number of days you want to predict int the future', '30')
    try:
        num_predictions = int(num_predictions_input)
    except ValueError:
        st.error("Please enter a valid integer.")
        num_predictions = 30  # default value if input is not valid

    # Load your model here
    # model = load_model(...)
    model = load_model('keras_model.h5')

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1_scaled = scaler.fit_transform(np.array(df.Close).reshape(-1, 1))

    # Assuming df1_scaled is your scaled input data
    input_data = scaler.fit_transform(final_df)

    # Generate predictions
    lst_output = generate_predictions(input_data, model, scaler, num_predictions)

    # Plotting
    st.subheader("Predicted Stock Prices")
    fig3 = plt.figure(figsize=(12, 6))
    day_new=np.arange(1,101)
    day_pred=np.arange(101,101+num_predictions)

    fig3=plt.figure(figsize=(12,6))
    plt.plot(day_new,scaler.inverse_transform(df1_scaled[(len(df1_scaled)-100):]))
    plt.plot(day_pred, scaler.inverse_transform(lst_output))
    st.pyplot(fig3)

    fig4 = plt.figure(figsize=(12,6))
    df3=df1_scaled.tolist()
    df3.extend(lst_output)
    df3=scaler.inverse_transform(df3).tolist()
    plt.plot(df3)
    st.pyplot(fig4)

# Add legend
plt.legend()


if __name__ == "__main__":
    main()

    




