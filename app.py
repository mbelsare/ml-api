import xgboost
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from PIL import Image
from io import BytesIO
import requests
import logging
import xgboost as xgb
from sklearn.datasets import make_classification

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# load pretrained XGBOOST model from model.json
# model_xgb = xgb.Booster()
# model_xgb.load_model("/models/model.json")

# load pre-trained model
model = ResNet50(weights='imagenet')


@app.route('/')
def ping():
    return 'Server is Up!'


@app.route('/predict', methods=['POST'])
def predict():
    # log the request
    logging.info(f'Request: {request.json}')

    # predict xgb model by creating ndaraay from test dataset
    # x,y = make_classification(50, 21, n_classes=2)
    # xtest, ytest = make_classification(10, 21, n_classes=2)
    # xgbtest = xgboost.DMatrix(xtest)
    # array = model_xgb.predict(xgbtest)
    # logging.info(array)

    # read image from request
    img_url = request.json['img_url']
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))

    # preprocess image
    img = img.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img[np.newaxis, ...])

    # make prediction
    preds = model.predict(img)
    preds = decode_predictions(preds, top=3)[0]

    # format prediction output
    result = []
    for pred in preds:
        result.append({'label': pred[1], 'probability': float(pred[2])})

    # log response
    logging.info(f'Response: {result}')

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
