# coding: utf-8

from flask import Flask, render_template, request, json
from sklearn.externals import joblib
import boto3
import numpy as np
import pickle
import dill

app = Flask(__name__)

BUCKET_NAME = 'ml-boston-housing'
MODEL_FNAME = 'model.pkl'

S3 = boto3.client('s3', region_name='us-east-1')

# memoized annotation, caches model file after it's pulled
def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
@app.route('/predict', methods=['POST'])
def make_prediction():
    app.logger.info("{} request received from: {}".format(
        request.method, request.remote_addr))

    if not request.get_json() is None:
        if 'data' in request.get_json():
            body = request.get_json()
            data = np.array([body['data']])

            result = {'result': predict(data)}
            return json.dumps(result)
        else:
            return json.dumps({'result': 'Error: '
                               'incoming json does not have data key field'})
    else:
        input_data = np.zeros((1, 13))
        for i, k in enumerate(request.form.keys()):
            input_data[0, i] = request.form.get(k, 0)

        result = predict(input_data)
        return render_template('index.html', label=result)


@memoize
def load_model(key):
    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
    modelstr = response['Body'].read()
    model = dill.loads(modelstr)

    return model  # joblib.load(key)

def predict(data):
    mdl = load_model(MODEL_FNAME)
    return str(np.squeeze(mdl.predict(data)))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
