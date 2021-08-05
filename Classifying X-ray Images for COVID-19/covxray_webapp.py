import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__) 

from keras.models import load_model
from keras.backend import set_session
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
print("Loading model")
global sess
sess = tf.Session()
set_session(sess)
global model
model = load_model('my_cifar10_model.h5')
global graph
graph = tf.get_default_graph()