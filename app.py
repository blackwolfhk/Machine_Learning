from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return """
    <h1>🎉 My Awesome ML App!</h1>
    <p>I just updated this through GitHub!</p>
    <p>CI/CD automatically deployed my changes!</p>
    """

@app.route('/health')
def health():
    return {'status': 'healthy', 'message': 'App is running!'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
