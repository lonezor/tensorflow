#!/usr/bin/env python

import sys
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from matplotlib.image import imread

parser = argparse.ArgumentParser()
parser.add_argument('-c', action='store', dest='path',
                    help='Classify image')
arguments = parser.parse_args()


def create_model():
	model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),
	tf.keras.layers.Dense(128, activation='relu'),
 	tf.keras.layers.Dropout(0.2),
 	tf.keras.layers.Dense(10, activation='softmax')
	])

	model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

	return model

def classify_path(x_test, y_test):
	model = create_model()
	model.load_weights('checkpoint_9')

	print("Opening image '" + arguments.path + "'")
	img = imread(arguments.path)

	x_test = x_test[0:9]
	y_test = y_test[0:9]

	for i in range(0,9):
		x_test[i] = img
		y_test[i] =  i+1
	loss,acc = model.evaluate(x_test,  y_test, verbose=2)
	predictions = model.predict(x_test)

	best_match = -1
	idx = -1
	i = 0
	for e in predictions[0]:
		print(e)
		if e > best_match:
			best_match = e
			idx = i
		i += 1
	
	print("Image match number " + str(idx))

def evaluate_against_test_data(x_test, y_test):
	print("Evaluate against official test data")
	for i in range(9,10):
		model = create_model()
		model.load_weights('checkpoint_' + str(i+1))
		loss,acc = model.evaluate(x_test,  y_test, verbose=2)
		model.summary()

def evaluate_against_own_data(x_test, y_test):
	print("Evaluate against own test data:\n")

	img_1 = imread('nr_1.png')
	img_2 = imread('nr_2.png')
	img_3 = imread('nr_3.png')
	img_4 = imread('nr_4.png')
	img_5 = imread('nr_5.png')
	img_6 = imread('nr_6.png')
	img_7 = imread('nr_7.png')
	img_8 = imread('nr_8.png')
	img_9 = imread('nr_9.png')

	# Reference image
	plt.imshow(x_test[0], cmap='Greys')
	#plt.show()

	# Own image
	plt.imshow(img_3, cmap='Greys')
	#plt.show()

	model = create_model()
	model.load_weights('checkpoint_9')

	x_test = x_test[0:1]
	y_test = y_test[0:1]

	img_arr = [img_1,img_2,img_3,img_4,img_5,img_6,img_7,img_8,img_9]

	for i in range(0,9):
		print("Testing nr #" + str(i+1))
		x_test = x_test[0:1]
		y_test = y_test[0:1]
		x_test[0] = img_arr[i]
		y_test[0] =  i+1
		loss,acc = model.evaluate(x_test,  y_test, verbose=2)
		print("\n")

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

if arguments.path != None:
	classify_path(x_test, y_test)
else:
	evaluate_against_test_data(x_test, y_test)
	evaluate_against_own_data(x_test, y_test)

	



