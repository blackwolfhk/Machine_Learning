from flask import Flask, render_template, request, jsonify
from ml_model import StockPredictor
import json

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

        if not symbol:
            return jsonify({'error': 'Please provide a valid stock symbol'}), 400

        # Get prediction
        prediction_result = predictor.predict_next_day(symbol)
        if prediction_result is None:
            return jsonify({'error': f'Unable to fetch data for {symbol}. Please check the symbol and try again.'}), 400

        # Get chart
        chart_json = predictor.create_chart(symbol)
        if chart_json is None:
            return jsonify({'error': f'Unable to create chart for {symbol}'}), 400

        chart_data = json.loads(chart_json)

        return jsonify({
            'prediction': prediction_result,
            'chart': chart_data,
            'symbol': symbol
        })

    except Exception as e:
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)