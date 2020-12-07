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

input_data = np.array(dataset['data']).astype('uint8')
output_data = np.array(dataset['labels']).astype('uint8')


n_spectrum = 3 # color chanel (RGB)
weight = 80
height = 80
X = input_data.reshape([-1, n_spectrum, weight, height])
X[0].shape



# get one chanel
pic = X[0]

rad_spectrum = pic[0]
green_spectrum = pic[1]
blue_spectum = pic[2]

# output encoding
y = to_categorical(output_data, 2)

# shuffle all indexes
indexes = np.arange(2800)
np.random.shuffle(indexes)

X_train = X[indexes].transpose([0,2,3,1])
y_train = y[indexes]


# normalization
X_train = X_train / 255

np.random.seed(42)

from tensorflow.keras import layers
from tensorflow.keras import initializers


he_initializer = initializers.HeNormal()

inputs = keras.Input(shape=(80, 80, 3), name="img")

x = layers.Conv2D(32, (3, 3), padding='same',activation='relu',kernel_initializer=he_initializer,
    bias_initializer="zeros")(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x) #40x40
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(32, (3, 3), padding='same',activation='relu',kernel_initializer=he_initializer,
    bias_initializer="zeros")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x) #20x20
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(64, (3, 3), padding='same',activation='relu',kernel_initializer=he_initializer,
    bias_initializer="zeros")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x) #10x10
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(64, (3, 3), padding='same',activation='relu',kernel_initializer=he_initializer,
    bias_initializer="zeros")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x) #5x5
x = layers.Dropout(0.25)(x)


x = layers.Conv2D(128, (3, 3), padding='same',activation='relu',kernel_initializer=he_initializer,
    bias_initializer="zeros")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x) #5x5
x = layers.Dropout(0.25)(x)


x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(2, activation='softmax')(x)
model = keras.Model(inputs, outputs, name="My_model")


from tensorflow.keras.utils import  plot_model as pm  #plotting the model structure
pm(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True,dpi=60)


# augmentation
# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# aug = ImageDataGenerator(

#   featurewise_center=True,
#   samplewise_center=True,
#   featurewise_std_normalization=True,
#   samplewise_std_normalization=True,
#   #zca_whitening=True,
#   #zca_epsilon=1e-06,
#   rotation_range=360,
#   width_shift_range=0.25,
#   height_shift_range=0.25,
#   brightness_range=(150,255),
#   shear_range=0.45,
#   zoom_range=0.35,
#   #channel_shift_range=0.35,
#   fill_mode="nearest",
#   #cval=0.0,
#   horizontal_flip=True,
#   vertical_flip=True,
#   rescale=0.35,
#   #preprocessing_function=None,
#   #data_format=None,
#   validation_split=0.35,
# )

aug = ImageDataGenerator(
  rotation_range=360,
  #zoom_range=0.2,
  width_shift_range=0.10,
  height_shift_range=0.10,
  #brightness_range=[0.7,1.0],
  shear_range=0.10,
  horizontal_flip=True,
  vertical_flip=True,
  fill_mode="nearest")

from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
# for storing logs into tensorboard
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")


callbacks = [
    ModelCheckpoint("./model_checkpoint", monitor='val_loss'),
    keras.callbacks.TensorBoard(log_dir=logdir)
]


# optimization setup
# sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
nadam = Nadam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07#, name="Nadam"#, **kwargs
)


model.compile(
    loss='categorical_crossentropy',
    optimizer=nadam, #sgd,
    metrics=['accuracy'])




# # training
# history = model.fit(
#     X_train,
#     y_train,
#     batch_size=32,
#     callbacks=callbacks,
#     epochs=18,
#     #steps_per_epoch=len(X_train) // 32,
#     validation_split=0.2,
#     shuffle=True,
#     verbose=1)

history = model.fit(
            x=aug.flow(X_train, y_train, batch_size=64),
            validation_data=(X_train, y_train),
            steps_per_epoch=len(X_train) // 64,
            callbacks=callbacks,
            epochs=5,
            verbose=1)

model.save('satseg5e_nadam.h5')
from keras.models import load_model
load_model('satseg5e_nadam.h5')

with open('history.json', 'w') as f:
    json.dump(history.history, f)

with open('history.json') as f:
    d = json.load(f)
    #print(d)


