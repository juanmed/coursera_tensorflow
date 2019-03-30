# Classifier for the MNIST dataset

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class nnCallback(keras.callbacks.Callback):
	
	# Callback to be executed after each epoch is completed
	def on_epoch_end(self, epoch, logs={}):
		
		# Finish training on acc > 0.99
		if(logs.get('acc') > 0.99):
			self.model.stop_traning = True
			print("Reached 99%. Training Terminated")

# load dataset
mnist = tf.keras.datasets.mnist
(train_img, train_lbl), (test_img, test_lbl) = mnist.load_data()

print("Train set: type {} shape {} max {}".format(type(train_img),train_img.shape,np.max(train_img)))
print("Labels: {}".format(np.unique(train_lbl)))

# Normalize 
train_img = train_img/255.0
test_img = test_img/255.0

# create network
model = keras.models.Sequential([keras.layers.Flatten(input_shape=(28,28)),
								 keras.layers.Dense(512, activation=tf.nn.sigmoid),
								 keras.layers.Dense(256, activation=tf.nn.relu),
								 #keras.layers.Dense(128, activation=tf.nn.relu),
								 keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compose network
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

train_interrupt = nnCallback()

# Train
model.fit(train_img, train_lbl, epochs=10, callbacks=[train_interrupt])

# Evalute test set
model.evaluate(test_img, test_lbl)