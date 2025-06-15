import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.utils
import json

class StockPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False

    def get_stock_data(self, symbol, period="3mo"):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if data.empty:
                return None
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def prepare_features(self, data):
        """Create features for ML model"""
        # Technical indicators
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['Price_Change'] = data['Close'].pct_change()
        data['Volume_Change'] = data['Volume'].pct_change()

        # Features for prediction
        features = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'Price_Change', 'Volume_Change']

        # Remove NaN values
        data = data.dropna()

        return data[features], data['Close']

    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def train_model(self, symbol):
        """Train the ML model on historical data"""
        data = self.get_stock_data(symbol, period="1y")
        if data is None:
            return False

        X, y = self.prepare_features(data)
        if len(X) < 10:  # Need enough data points
            return False

        # Prepare training data (predict next day's closing price)
        X_train = X[:-1]  # All but last day
        y_train = y[1:]  # Next day's price

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        return True

    def predict_next_day(self, symbol):
        """Predict next day's closing price"""
        if not self.is_trained:
            if not self.train_model(symbol):
                return None

        # Get latest data
        data = self.get_stock_data(symbol, period="3mo")
        if data is None:
            return None

        X, y = self.prepare_features(data)
        if len(X) == 0:
            return None

        # Use latest day for prediction
        latest_features = X.iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)

        # Make prediction
        prediction = self.model.predict(latest_features_scaled)[0]
        current_price = y.iloc[-1]

        return {
            'current_price': round(current_price, 2),
            'predicted_price': round(prediction, 2),
            'change': round(prediction - current_price, 2),
            'change_percent': round(((prediction - current_price) / current_price) * 100, 2)
        }

    def create_chart(self, symbol):
        """Create interactive price chart"""
        data = self.get_stock_data(symbol, period="3mo")
        if data is None:
            return None

        fig = go.Figure()

        # Add price line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Stock Price',
            line=dict(color='#1f77b4', width=2)
        ))

        # Add moving averages
        sma_20 = data['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=sma_20,
            mode='lines',
            name='20-day SMA',
            line=dict(color='orange', width=1, dash='dash')
        ))

        fig.update_layout(
            title=f'{symbol} Stock Price (Last 3 Months)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)