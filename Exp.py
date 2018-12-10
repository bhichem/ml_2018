# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 11:30:54 2018

@author: Amey
"""
import skimage

import numpy as np
import os
import time
import cv2
from resnet50 import ResNet50
from vgg16 import VGG16
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.utils import np_utils, to_categorical
from sklearn import preprocessing
from keras import optimizers
from keras.engine import Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace


PATH = os.getcwd()
data_path = PATH + '/Dataset_resized'
data_dir_list = os.listdir(data_path)

img_data_list = []
labels = []
folderCounter = 0

def flip_images(image):
    flipped_image = np.fliplr(image)
    return flipped_image


def rotate(image, angle):
    rotated_image = skimage.transform.rotate(image, angle= angle, preserve_range=True).astype(np.uint8)
    return rotated_image


def gaussian_noise(image):
    mean = 20;
    std = 20;
    noisy_img = image + np.random.normal(mean, std, image.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    return noisy_img_clipped


def lighting(image, gamma):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   flippedLighting = cv2.LUT(image, table)
   return flippedLighting

list_of_image_paths = []
labelTest = []

for eachfolderName in data_dir_list:
    # print(eachfolderName)
    img_list = os.listdir(data_path + '/' + eachfolderName)

    for image in img_list:
        retrieve_dir = data_path + "\\" + eachfolderName + "\\" + image
        images = cv2.imread(retrieve_dir, 3)
        list_of_image_paths.append(images)

    img_data_list.append(img_list)
    labels.append(eachfolderName)

    labelTest.append([eachfolderName] * len(img_list))
    imageCounter = 0

    folderCounter = folderCounter + 1;

list_of_image_paths = np.array(list_of_image_paths)
flattened_list = [y for x in labelTest for y in x]
flattened_list = np.asarray(flattened_list, dtype="str")
lb = preprocessing.LabelBinarizer()
lb.fit(flattened_list)
lb.classes_
labels = lb.transform(flattened_list)

X_train, X_test, y_train, y_test = train_test_split(list_of_image_paths, labels, test_size=0.2, random_state=2)

augmented_Xtrain = []
y_train_labels = []

images = []
for index in range(len(X_train)):


    image = X_train[index]
    label = y_train[index]

    images.append(image)
    images.append(flip_images(image))
    images.append(rotate(image, 10))
    images.append(rotate(image,  -10))
    images.append(gaussian_noise(image))
    images.append(lighting(image, 0.5))
    images.append(lighting(image,  1.5))
   #+=[image, flipped_image, rotatedMinus10Image, rotatedPlus10Image, gaussianImage, darkerImage, lighterImage]

    y_train_labels.extend([label]*7)

augmented_X_train = np.uint8(images)
y_train_labels = np.int32(y_train_labels)
#################VGGFace Model############

image_input = Input(shape=(224, 224, 3))
nb_class = 136
hidden_dim = 2048

vggface_model = VGGFace(input_tensor=image_input, include_top=True, weights='vggface', input_shape=(224, 224, 3),
                         pooling='max')
vggface_model.summary()

pool5 = vggface_model.layers[-8]

flatten = vggface_model.layers[-7]

fc6 = vggface_model.layers[-6]
fc6relu = vggface_model.layers[-5]

fc7 = vggface_model.layers[-4]
fc7relu = vggface_model.layers[-3]

fc8 = vggface_model.layers[-2]
fc8relu = vggface_model.layers[-1]

dropout1 = Dropout(0.5)
dropout2 = Dropout(0.5)

# Reconnect the layers
x = dropout1(pool5.output)
x = flatten(x)
x = fc6(x)
x = fc6relu(x)
x = dropout2(x)
x = fc7(x)
x = fc7relu(x)
out = Dense(136, activation='softmax', name='output')(x)

custom_vggface_model = Model(vggface_model.input, output=out)
 # Resnet_model = ResNet50(weights='imagenet',include_top=False)


vggface_model.summary()
custom_vggface_model.summary()

for layer in custom_vggface_model.layers[:-1]:
     layer.trainable = False

custom_vggface_model.layers[-1].trainable
custom_vggface_model.summary()

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
custom_vggface_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

t = time.time()
hist = custom_vggface_model.fit(augmented_X_train, y_train_labels, batch_size=16, epochs=200, verbose=1,
                                 validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vggface_model.evaluate(X_test, y_test, batch_size=16, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

#################VGG Model############
# image_input = Input(shape=(224, 224, 3))
#
# vgg_model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
# vgg_model.summary()
# last_layer = vgg_model.get_layer('fc2').output
##x= Flatten(name='flatten')(last_layer)
# out = Dense(136, activation='softmax', name='output')(last_layer)
# custom_vgg_model = Model(image_input, out)
# custom_vgg_model.summary()
#
# for layer in custom_vgg_model.layers[:-1]:
#	layer.trainable = False
#
# custom_vgg_model.layers[3].trainable
# custom_vgg_model.summary()
# sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
# custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='adam' ,metrics=['accuracy'])
#
#
# t=time.time()
##	t = now()
# hist = custom_vgg_model.fit(X_train, y_train, batch_size=20, epochs=20, verbose=1, validation_data=(X_test, y_test))
# print('Training time: %s' % (t - time.time()))
# (loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=20, verbose=1)
#
# print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


