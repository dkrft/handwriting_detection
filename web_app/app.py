# Load the packages

from flask import Flask, render_template, request
import os

# Connect the app
app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'tmp'


@app.route('/')
def homepage():
    """Render home page"""
    return render_template('home.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if not os.path.isdir(app.config["UPLOAD_FOLDER"]):
        os.mkdir(app.config["UPLOAD_FOLDER"])
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    # add your custom code to check that the uploaded file is a valid image
    # and not a malicious file (out-of-scope for this post)
    file.save(f)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)  # Set to false when deploying
