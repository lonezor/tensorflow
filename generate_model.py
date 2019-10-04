#!/usr/bin/env python

import tensorflow as tf

print("Download MNIST data")
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print("Adjust image data")
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Build model")
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

for i in range(0,10):
	print("Process training data, iteration #" + str(i+1))
	model.fit(x_train, y_train, epochs=1)
	model.save_weights('checkpoint_' + str(i+1))



