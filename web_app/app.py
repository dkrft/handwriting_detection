# Load the packages

from flask import Flask, render_template, request

# Connect the app
app = Flask(__name__)


@app.route('/')
def homepage():
    """Render home page"""
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)  # Set to false when deploying
