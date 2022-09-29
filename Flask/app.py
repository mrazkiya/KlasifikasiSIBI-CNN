from __future__ import division, print_function
from flask import Flask, render_template

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# defineapp
app = Flask(__name__)

# modelimport
MODEL_PATH = 'models/e60.h5'

# load
model = load_model(MODEL_PATH)


print('Model loaded. Check http://127.0.0.1:3000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64), grayscale=True)

    # Preprocessing the image
    # img = cv2.imread(dataset_path + "/" +x)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize = cv2.resize(img, dsize=(64, 64))
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    print(preds)
    return preds


def kelas(predict):
    kelas = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
             "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    for i, x in enumerate(predict[0]):
        if x >= 0.6:
            return kelas[i]
        else:
            return ("Gambar bukan abjad SIBI")


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = {'predict': kelas(preds)}               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
