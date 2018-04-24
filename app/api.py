# coding: utf-8

from flask import Flask, render_template, request, jsonify
from sklearn.externals import joblib
import boto3
import numpy as np
import pickle

app = Flask(__name__)

BUCKET_NAME = 'ml-boston-housing'
MODEL_FNAME = 'model.pkl'

S3 = boto3.client('s3', region_name='us-east-1')


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        app.logger.info("{} request received from: {}".format(
            request.method, request.remote_addr))

        body = request.get_json()
        if type(body) == 'dict':
            data = body['data']
            print('\nrequest json output:\n\n', data)
        else:
            print('\nrequest json output:\n\n', body)

        mdl = load_model(MODEL_FNAME)

        if not request.json() is None:
            if 'data' in request.json():
                body = request.get_json()
                data = body['data']
                print('POST RECIEVED JSON DATA:\n\tData Type:{}\n\tData Shape: '.format(
                    type(data), data.shape))
                result = np.squeeze(mdl.predict(data))
                return jsonify({'result': result})
        else:
            input_data = np.zeros((1, 13))
            for i, k in enumerate(request.form.keys()):
                input_data[0, i] = request.form.get(k, 0)

            result = np.squeeze(mdl.predict(input_data))
            return render_template('index.html', label=result)


# memoized annotation, caches model file after it's pulled
def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper


@memoize
def load_model(key):
    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
    print('response : ', type(response['Body'].read()))
    # modelstr = response['Body'].read()
    # model = pickle.loads(modelstr)

    return joblib.load('model.pkl')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
