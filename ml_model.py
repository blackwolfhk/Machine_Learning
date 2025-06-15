import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.utils
import json
import requests
import time
import random
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.session = None
        self.last_request_time = 0
        self.min_request_interval = 2  # Minimum seconds between requests

    def get_session(self):
        """Create a session with proper headers to avoid rate limiting"""
        if self.session is None:
            self.session = requests.Session()
            # Use realistic browser headers
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            })
        return self.session

    def rate_limit_delay(self):
        """Ensure minimum delay between API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            delay = self.min_request_interval - time_since_last
            logger.info(f"Rate limiting: waiting {delay:.1f} seconds...")
            time.sleep(delay)

        self.last_request_time = time.time()

    def test_internet_connection(self):
        """Test if internet connection is available"""
        try:
            response = requests.get('https://httpbin.org/ip', timeout=5)
            logger.info(f"✅ Internet connection: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ Internet connection failed: {e}")
            return False

    def get_stock_data(self, symbol, period="3mo", retries=3):
        """Fetch stock data with smart rate limiting and fallback"""

        # Check internet first
        if not self.test_internet_connection():
            logger.warning("No internet connection, using mock data")
            return self.generate_mock_data(symbol)

        for attempt in range(retries):
            try:
                logger.info(f"🔄 Fetching {symbol} data (attempt {attempt + 1}/{retries})")

                # Rate limiting delay
                self.rate_limit_delay()

                # Add random delay to avoid pattern detection
                time.sleep(random.uniform(0.5, 1.5))

                # Try different approaches
                if attempt == 0:
                    # Method 1: Direct yfinance with session
                    stock = yf.Ticker(symbol, session=self.get_session())
                elif attempt == 1:
                    # Method 2: Simple yfinance without session
                    stock = yf.Ticker(symbol)
                else:
                    # Method 3: Force new session
                    self.session = None
                    stock = yf.Ticker(symbol, session=self.get_session())

                # Try to get data
                data = stock.history(period=period, interval="1d")

                if data.empty:
                    logger.warning(f"Empty data for {symbol} on attempt {attempt + 1}")
                    if attempt < retries - 1:
                        time.sleep(3)  # Longer delay for empty data
                        continue
                    else:
                        logger.info(f"Using mock data for {symbol} after {retries} attempts")
                        return self.generate_mock_data(symbol)

                logger.info(f"✅ Successfully fetched {len(data)} records for {symbol}")
                return data

            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    logger.warning(f"Rate limited for {symbol}, attempt {attempt + 1}")
                    delay = (attempt + 1) * 5  # Exponential backoff: 5, 10, 15 seconds
                    logger.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    logger.error(f"HTTP error for {symbol}: {e}")

            except Exception as e:
                logger.error(f"Error fetching {symbol} on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(2)

        # All attempts failed, use mock data
        logger.info(f"All attempts failed for {symbol}, using mock data")
        return self.generate_mock_data(symbol)

    def generate_mock_data(self, symbol):
        """Generate realistic mock stock data"""
        logger.info(f"🎭 Generating mock data for {symbol}")

        # Realistic base prices for major stocks
        base_prices = {
            'AAPL': 175.0, 'GOOGL': 140.0, 'MSFT': 350.0, 'TSLA': 200.0,
            'AMZN': 145.0, 'NVDA': 800.0, 'META': 300.0, 'NFLX': 400.0,
            'AMD': 140.0, 'ORCL': 100.0, 'CRM': 200.0, 'INTC': 30.0,
            'SPY': 450.0, 'QQQ': 380.0, 'IWM': 200.0, 'VTI': 240.0
        }

        base_price = base_prices.get(symbol.upper(), 100.0)

        # Generate 90 days of realistic data
        dates = pd.date_range(end=datetime.now().date(), periods=90, freq='D')

        # Use symbol hash for consistent data
        np.random.seed(hash(symbol.upper()) % (2 ** 31))

        # Generate realistic stock movements
        returns = np.random.normal(0.0005, 0.015, 90)  # Small positive bias, 1.5% volatility
        prices = [base_price]

        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            # Add some constraints to keep prices realistic
            if new_price < base_price * 0.7:  # Don't drop below 70% of base
                new_price = prices[-1] * (1 + abs(ret) * 0.3)
            elif new_price > base_price * 1.5:  # Don't go above 150% of base
                new_price = prices[-1] * (1 - abs(ret) * 0.3)
            prices.append(new_price)

        # Create realistic OHLCV data
        data_rows = []
        for i, close_price in enumerate(prices):
            # Generate intraday range
            daily_range = abs(np.random.normal(0, 0.01))
            high = close_price * (1 + daily_range)
            low = close_price * (1 - daily_range)

            # Open is usually close to previous close
            if i > 0:
                gap = np.random.normal(0, 0.005)  # Small overnight gaps
                open_price = prices[i - 1] * (1 + gap)
            else:
                open_price = close_price

            # Ensure OHLC relationship is valid
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            # Realistic volume (varies by stock)
            base_volume = {
                'AAPL': 80000000, 'GOOGL': 25000000, 'MSFT': 40000000,
                'TSLA': 60000000, 'SPY': 100000000, 'QQQ': 50000000
            }.get(symbol.upper(), 30000000)

            volume = int(base_volume * np.random.uniform(0.3, 2.0))

            data_rows.append({
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close_price, 2),
                'Volume': volume
            })

        df = pd.DataFrame(data_rows, index=dates)
        logger.info(f"✅ Generated {len(df)} realistic mock records for {symbol}")
        return df

    def prepare_features(self, data):
        """Create technical indicators for ML model"""
        df = data.copy()

        # Technical indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open']

        # Features for prediction
        features = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_20',
                    'RSI', 'Price_Change', 'Volume_Change', 'High_Low_Pct', 'Open_Close_Pct']

        # Remove NaN values
        df = df.dropna()

        return df[features], df['Close']

    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI

    def train_model(self, symbol):
        """Train the ML model"""
        try:
            logger.info(f"🧠 Training model for {symbol}")
            data = self.get_stock_data(symbol, period="1y")

            if data is None or len(data) < 30:
                logger.error(f"Insufficient data for {symbol}")
                return False

            X, y = self.prepare_features(data)
            if len(X) < 20:
                logger.error(f"Not enough features for {symbol}")
                return False

            # Prepare training data (predict next day's price)
            X_train = X[:-1]  # All but last day
            y_train = y[1:]  # Next day's price

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Train model
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True

            logger.info(f"✅ Model trained successfully for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Model training failed for {symbol}: {e}")
            return False

    def predict_next_day(self, symbol):
        """Make next-day price prediction"""
        try:
            if not self.is_trained:
                if not self.train_model(symbol):
                    return None

            # Get latest data for prediction
            data = self.get_stock_data(symbol, period="3mo")
            if data is None or len(data) == 0:
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

            result = {
                'current_price': round(float(current_price), 2),
                'predicted_price': round(float(prediction), 2),
                'change': round(float(prediction - current_price), 2),
                'change_percent': round(float((prediction - current_price) / current_price * 100), 2)
            }

            logger.info(f"✅ Prediction for {symbol}: {result}")
            return result

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None

    def create_chart(self, symbol):
        """Create price chart with predictions"""
        try:
            data = self.get_stock_data(symbol, period="3mo")
            if data is None or len(data) == 0:
                return None

            fig = go.Figure()

            # Main price line
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name=f'{symbol} Price',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))

            # Moving averages
            sma_20 = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=sma_20,
                mode='lines',
                name='20-day Average',
                line=dict(color='orange', width=1, dash='dash'),
                hovertemplate='Date: %{x}<br>SMA20: $%{y:.2f}<extra></extra>'
            ))

            # Chart title with data source info
            title = f'{symbol} Stock Price - Last 3 Months'
            if not self.test_internet_connection():
                title += ' (Demo Data)'

            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                template='plotly_white',
                height=400,
                showlegend=True
            )

            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        except Exception as e:
            logger.error(f"Chart creation failed for {symbol}: {e}")
            return None