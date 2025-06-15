from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return """
    <h1>🎉 Automatic CI/CD is Working!</h1>
    <p>I just pushed this to GitHub and Jenkins auto-deployed it!</p>
    <p>No manual clicking needed!</p>
    """

@app.route('/health')
def health():
    return {'status': 'healthy', 'message': 'App is running!'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
