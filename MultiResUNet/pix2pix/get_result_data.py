import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras import backend as K 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import glob


#%tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


import glob
all_img_files = glob.glob('maps_ds/train_images/*')
all_mask_files = glob.glob('maps_ds/train_mask/*')
print(len(all_img_files))
print(len(all_mask_files))
print(all_img_files[:10])
print(all_mask_files[:10])

#img = cv2.imread('trainx/X_img_144.bmp', cv2.IMREAD_COLOR)
#img.shape

#plt.imshow(img[:,:,::-1])

img_files = glob.glob('maps_ds/train_images/*')
msk_files = glob.glob('maps_ds/train_mask/*')

img_files.sort()
msk_files.sort()

print(len(img_files))
print(len(msk_files))


X = []
Y = []

for img_fl in tqdm(img_files):
  #print(img_fl)
  #break
  img = cv2.imread('{}'.format(img_fl), cv2.IMREAD_COLOR)
  resized_img = cv2.resize(img,(256, 256), interpolation = cv2.INTER_CUBIC)
  #plt.imshow(resized_img)
  #plt.show()
  X.append(resized_img)
  mask_name = 'maps_ds/train_mask/'+str(img_fl.split('.')[0]).split('/')[-1]+".jpg"
  #print("mn = ",mask_name)
  #break
  msk = cv2.imread('{}'.format(mask_name), cv2.IMREAD_GRAYSCALE)
  resized_msk = cv2.resize(msk,(256, 256), interpolation = cv2.INTER_CUBIC)
  #plt.imshow(resized_msk)
  Y.append(resized_msk)
  #break
print(len(X))
print(len(Y))



X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

X_train_name, X_test_name, Y_train_name, Y_test_name = train_test_split(img_files, msk_files, test_size=0.2, random_state=3)



Y_train = Y_train.reshape((Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))
Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))

X_train = X_train / 255
X_test = X_test / 255
Y_train = Y_train / 255
Y_test = Y_test / 255

Y_train = np.round(Y_train,0)
Y_test = np.round(Y_test,0)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)





