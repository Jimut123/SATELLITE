import json, sys, random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from keras.optimizers import SGD, Nadam
import keras.callbacks

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# download dataset from json object
f = open(r'shipsnet.json')
dataset = json.load(f)
f.close()


