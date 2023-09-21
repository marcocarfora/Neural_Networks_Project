# Import needed libraries
import numpy as np
import tensorflow as tf
import pandas as pd

import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers, optimizers, callbacks
from tensorflow.keras.losses import categorical_crossentropy

import cv2
import requests
import datetime
import os
from google.colab import drive
drive.mount('/content/drive')	# Connect with Google Drive

import matplotlib.pyplot as plt
import sklearn.metrics 
from sklearn.metrics import classification_report, confusion_matrix
import pylab as pl


# Print TensorFlow and Keras versions
print("Tensorflow version %s" %tf.__version__)
print("Keras version %s" %keras.__version__)


# Force to use the GPU, otherise training would take too much time
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0': raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
print("\n---------------------------------------------------------------\n")

