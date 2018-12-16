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
#from resnet50 import ResNet50
from vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16

from keras.preprocessing import image as Image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.utils import np_utils, to_categorical
from sklearn import preprocessing
from keras import optimizers
from keras.engine import Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer
from keras.applications.imagenet_utils import preprocess_input
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
#from imagenet_utils import decode_predictions




nb_classes = 136
nb_epochs = 10
b_size = 16
augment_data = False
PATH = os.getcwd()
data_path = PATH + '/Dataset'
data_dir_list = os.listdir(data_path)
plot_data = False
training_model = 3

mlb = MultiLabelBinarizer()

img_data_list = []
labels = []
list_of_image_paths = []
label_test = []

for folder_name in data_dir_list:
	img_list = os.listdir(data_path + '/' + folder_name)

	for image in img_list:
		retrieve_dir = data_path + "/" + folder_name + "/" + image
		images = cv2.imread(retrieve_dir, 3)
		list_of_image_paths.append(images)

	img_data_list.append(img_list)
	labels.append(folder_name)

	label_test.append([folder_name] * len(img_list))


list_of_image_paths = np.array(list_of_image_paths)
flattened_list = [y for x in label_test for y in x]
flattened_list = np.asarray(flattened_list, dtype="str")
lb = preprocessing.LabelBinarizer()
lb.fit(flattened_list)
lb.classes_
labels = lb.transform(flattened_list)
#labels = mlb.fit_transform(flatten_list)
print(len(labels))

X_train, X_test, y_train, y_test = train_test_split(list_of_image_paths, labels, test_size=0.2, random_state=2)

augmented_Xtrain = []
y_train_labels = []

images = []
train_datagen = ImageDataGenerator(featurewise_std_normalization=True, rotation_range=10, horizontal_flip=True, vertical_flip=True, zoom_range=0.4)

for index in range(len(X_train)):

	img = X_train[index]
	label = y_train[index]

	images.append(img)
	if augment_data:
		img = np.expand_dims(img,0)
		augmented_itr = train_datagen.flow(img)
		[images.append(next(augmented_itr)[0].astype(np.uint8)) for i in range(6)]
		y_train_labels.extend([label]*7)
	else:
		y_train_labels.append(label)

augmented_X_train = np.uint8(images)
y_train_labels = np.int32(y_train_labels)

# training_model 1 for VGGFace 2 for VGG16
im_shape = (224,224,3)
image_input = Input(shape=im_shape)

#################VGGFace Model############
if training_model == 1:

	hidden_dim = 2048

	vggface_model = VGGFace(input_tensor=image_input, include_top=True, weights='vggface', input_shape=im_shape,
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
	out = Dense(CLASSESNR, activation='softmax', name='output')(x)

	custom_vggface_model = Model(vggface_model.input, output=out)
	 # Resnet_model = ResNet50(weights='imagenet',include_top=False)

	vggface_model.summary()
	custom_vggface_model.summary()

	for layer in custom_vggface_model.layers[:-1]:
		 layer.trainable = False

	custom_vggface_model.layers[-1].trainable
	custom_vggface_model.summary()

	sgd = optimizers.SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
	custom_vggface_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	t = time.time()
	hist = custom_vggface_model.fit(augmented_X_train, y_train_labels, batch_size= b_size, epochs=nb_epochs, verbose=1,
		                             validation_data=(X_test, y_test))
	print('Training time: %s' % (t - time.time()))
	(loss, accuracy) = custom_vggface_model.evaluate(X_test, y_test, batch_size= b_size, verbose=1)

	print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

#################VGG Model############
if training_model == 2:
	vgg_model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
	vgg_model.summary()
	last_layer = vgg_model.get_layer('fc2').output
	##x= Flatten(name='flatten')(last_layer)
	out = Dense(nb_classes, activation='softmax', name='output')(last_layer)
	custom_vgg_model = Model(image_input, out)
	custom_vgg_model.summary()

	for layer in custom_vgg_model.layers[:-1]:
		layer.trainable = False
	custom_vgg_model.layers[3].trainable
	custom_vgg_model.summary()
	sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='adam' ,metrics=['accuracy'])

	t=time.time()
	t = now()

	hist = custom_vgg_model.fit(X_train, y_train, batch_size = b_size, epochs = nb_epochs, verbose=1, validation_data=(X_test, y_test))
	print('Training time: %s' % (t - time.time()))
	(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=20, verbose=1)

	print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


if training_model == 3:

	model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
	model.summary()
	last_layer = model.get_layer('avg_pool').output
	#x= Flatten(name='flatten')(last_layer)
	out = Dense(nb_classes, activation='softmax', name='output_layer')(last_layer)
	custom_resnet_model = Model(inputs=image_input,outputs= out)
	custom_resnet_model.summary()

	for layer in custom_resnet_model.layers[:-1]:
		layer.trainable = False

	custom_resnet_model.layers[-1].trainable
	sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
	custom_resnet_model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy', 'binary_crossentropy'])


	t=time.time()
	hist = custom_resnet_model.fit(augmented_X_train, y_train_labels, batch_size=b_size, epochs=nb_epochs, verbose=2, validation_data=(X_test, y_test))
	print('Training time: %s' % (t - time.time()))
	#(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=32, verbose=1)

	#print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

	for test_img, test_label in zip(X_test, y_test):
		x = Image.img_to_array(test_img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		prediction = custom_resnet_model.predict(x)
		index = np.argmax(prediction) 
		index2 = np.argmax(test_label) 
		print(prediction, test_label)
		print(index,index2, prediction[0][index], test_label[index], prediction[0][index2])
	
	if plot_data:
		polot_daa(hist)



def plot_data(hist):
	# visualizing losses and accuracy
	train_loss = hist.history['loss']
	val_loss = hist.history['val_loss']
	train_acc = hist.history['acc']
	val_acc = hist.history['val_acc']
	xc=range(12)

	plt.figure(1,figsize=(7,5))
	plt.plot(xc,train_loss)
	plt.plot(xc,val_loss)
	plt.xlabel('num of Epochs')
	plt.ylabel('loss')
	plt.title('train_loss vs val_loss')
	plt.grid(True)
	plt.legend(['train','val'])
	#print plt.style.available # use bmh, classic,ggplot for big pictures
	plt.style.use(['classic'])

	plt.figure(2,figsize=(7,5))
	plt.plot(xc,train_acc)
	plt.plot(xc,val_acc)
	plt.xlabel('num of Epochs')
	plt.ylabel('accuracy')
	plt.title('train_acc vs val_acc')
	plt.grid(True)
	plt.legend(['train','val'],loc=4)
	#print plt.style.available # use bmh, classic,ggplot for big pictures
	plt.style.use(['classic'])
	plt.savefig('test.png')



	







