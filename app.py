from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return '''
    <h1>🚀 Machine Learning App Deployed!</h1>
    <p>Your CI/CD pipeline is working correctly!</p>
    <p>Jenkins successfully deployed this app to EC2.</p>
    '''

@app.route('/health')
def health():
    return {'status': 'healthy', 'message': 'App is running!'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
