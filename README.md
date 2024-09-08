
# Stock Price Prediction using LSTM

## Overview
This project demonstrates stock price prediction using the Long Short-Term Memory (LSTM) model in Python. The model is trained on Apple Inc.'s stock price data, retrieved using Yahoo Finance (`yfinance`). The objective is to forecast future stock prices based on historical data using machine learning.

## Features
- Fetches stock price data using Yahoo Finance (`yfinance`).
- Preprocesses and scales the data using `scikit-learn`'s `MinMaxScaler`.
- Implements an LSTM model with TensorFlow/Keras to predict stock prices.
- Visualizes training and prediction results using Matplotlib.
- Evaluates model performance using Root Mean Square Error (RMSE).

## Technologies
- **Python**
- **TensorFlow / Keras**
- **Pandas**
- **Numpy**
- **Yahoo Finance API (`yfinance`)**
- **Matplotlib**
- **Scikit-learn**

## Setup and Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction-lstm.git
   cd stock-price-prediction-lstm
   ```

2. **Create and activate a virtual environment**:
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the script**:
   ```bash
   python "Stock price.py"
   ```

## License
This project is licensed under the MIT License.
