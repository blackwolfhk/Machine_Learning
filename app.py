from flask import Flask, render_template, request, jsonify
from ml_model import StockPredictor
import json
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the ML model
predictor = StockPredictor()

@app.route('/')
def home():
    """Main page with stock prediction interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for stock prediction"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()

        logger.info(f"Prediction request for symbol: {symbol}")

        if not symbol:
            logger.warning("Empty symbol provided")
            return jsonify({'error': 'Please provide a valid stock symbol'}), 400

        # Validate symbol format
        if not symbol.isalpha() or len(symbol) > 10:
            logger.warning(f"Invalid symbol format: {symbol}")
            return jsonify({'error': f'Invalid symbol format: {symbol}'}), 400

        logger.info(f"Attempting to get prediction for {symbol}")

        # Get prediction
        prediction_result = predictor.predict_next_day(symbol)
        if prediction_result is None:
            logger.error(f"Prediction failed for {symbol}")
            return jsonify({'error': f'Unable to fetch data for {symbol}. Please check the symbol and try again.'}), 400

        logger.info(f"Prediction successful for {symbol}: {prediction_result}")

        # Get chart
        chart_json = predictor.create_chart(symbol)
        if chart_json is None:
            logger.error(f"Chart creation failed for {symbol}")
            return jsonify({'error': f'Unable to create chart for {symbol}'}), 400

        chart_data = json.loads(chart_json)

        logger.info(f"Successfully processed request for {symbol}")

        return jsonify({
            'prediction': prediction_result,
            'chart': chart_data,
            'symbol': symbol
        })

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'ML Stock Predictor',
        'version': '1.0.0'
    })

@app.route('/api/test')
def api_test():
    """Simple test endpoint"""
    return jsonify({
        'message': 'API is working!',
        'endpoints': [
            '/health - Health check',
            '/predict - Stock prediction (POST)',
            '/api/test - This test endpoint'
        ]
    })

@app.route('/debug/<symbol>')
def debug_symbol(symbol):
    """Debug endpoint to test stock data fetching"""
    try:
        import yfinance as yf

        logger.info(f"Debug request for symbol: {symbol}")

        # Test basic connectivity
        stock = yf.Ticker(symbol.upper())
        info = stock.info

        # Test data fetching
        data = stock.history(period="5d")

        return jsonify({
            'symbol': symbol.upper(),
            'info_available': bool(info),
            'data_points': len(data),
            'data_sample': data.tail(2).to_dict() if len(data) > 0 else None,
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Debug error for {symbol}: {str(e)}")
        return jsonify({
            'symbol': symbol.upper(),
            'error': str(e),
            'traceback': traceback.format_exc(),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)