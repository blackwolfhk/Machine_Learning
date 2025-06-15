from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return """
    <h1>🎉 Automatic CI/CD is Working!</h1>
    <p>I just pushed this to GitHub and Jenkins auto-deployed it!</p>
    <p>No manual clicking needed!</p>
    """
    # Missing closing quote will break Python syntax

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000