# Stock Trend Predictor App

## Overview

The Stock Trend Predictor App is a web application that utilizes a Long Short-Term Memory (LSTM) neural network to predict stock prices. The app is built using TensorFlow and Keras for the machine learning model, with data handling and visualization facilitated by pandas, numpy, yfinance, and matplotlib. The app's interface is created using Streamlit, making it easy to interact with the prediction model.

## Features

- Fetch stock price data from Yahoo Finance.
- Preprocess the data for model training and prediction.
- Train an LSTM model on the stock price data.
- Visualize historical stock prices and model predictions.
- Provide future stock price predictions.

## Installation

To run the Stock Trend Predictor App, you need to have Python installed on your machine. Follow the steps below to set up the environment and run the application.

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/arnavdaryani/StockPredictor.git
cd StockPredictor
```
### Install Dependencies

```bash
pip install tensorflow keras pandas numpy yfinance matplotlib scikit-learn streamlit
```

### Usage

```bash
streamlit run app.py
```

### App Interface

- Stock Ticker Input: Enter the ticker symbol of the stock you want to predict (e.g., AAPL for Apple Inc.).
- Data Visualization: View the historical stock prices in a line chart.
- Model Training: Train the LSTM model using the historical data.
- Prediction Visualization: View the predicted stock prices alongside the historical data.
