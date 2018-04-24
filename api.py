import flask
from flask import Flask, render_template, request
from sklearn.externals import joblib

app = Flask(__name__)


@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


# create endpoint for the predictions (HTTP POST requests)
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        file = request.files['housing-data']

        if not file:
            return render_template('index.html', label='No File')

        return render_template('index.html', label='3')


if __name__ == '__main__':
    # LOAD MODEL WHEN APP RUNS ####
    model = joblib.load('model.pkl')
    app.run(host='0.0.0.0', port=8000, debug=True)
