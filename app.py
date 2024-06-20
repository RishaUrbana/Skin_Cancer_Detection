# load model


# from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, keras

import pickle
import joblib
import sys, types
import numpy as np
import tensorflow as tf
import streamlit as st

from keras  import backend as K
model = tf.keras.models.load_model('exportedModel.h5', compile=False)  # Assuming the model was saved in HDF5 format


K.clear_session()




from PIL import Image
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img
from keras._tf_keras.keras.preprocessing.image import img_to_array
from keras._tf_keras.keras.applications.imagenet_utils import preprocess_input
from keras._tf_keras.keras.applications.imagenet_utils import decode_predictions



from tensorflow.python.keras.saving.hdf5_format import save_attributes_to_hdf5_group # https://github.com/huggingface/transformers/issues/18912



from flask import Flask, render_template, request
app = Flask(__name__, template_folder="./", static_url_path="", static_folder="")

@app.route('/', methods=['GET'])
def homepage():
	return render_template('index.html')

@app.route('/generator', methods=['POST','GET'])
def generator():
	print("insdie generator fun.")
	imagefile = request.files['imagefile']
	image_path = './image_submitted/' + imagefile.filename
	print("image_path")
	imagefile.save(image_path)

	 # preprocessing
	image = load_img(image_path, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	print(type(image))
	yhat = model.predict(image)
	label = decode_predictions(yhat)
	label = label[0][0]

	classification = "%s (%.2f%%)" % (label[1], label[2]*100)
	
	print("prection result:.........", label)
	return render_template('generator.html')

if __name__ == "__main__":

	app.run(port=3000, debug=True)
	 