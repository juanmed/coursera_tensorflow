import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
import time


# **************************************************
#   FIX for conv networks
#   https://github.com/tensorflow/tensorflow/issues/24828
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# ***************************************************


# get fashion dataset
fmnist = keras.datasets.fashion_mnist

# get train, tests sets
(train_img, train_lbl), (test_img, test_lbl) = fmnist.load_data()

# Reshape and Normalize
train_img = train_img.reshape(60000, 28, 28, 1)/255.0
test_img = test_img.reshape(10000, 28, 28, 1)/255.0

# create network
model = keras.models.Sequential([
	keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)),
	keras.layers.MaxPooling2D(2,2),
	keras.layers.Conv2D(8, (3,3), activation='relu'),
	keras.layers.MaxPooling2D(2,2),
	keras.layers.Flatten(),
	keras.layers.Dense(128, activation = 'relu'),
	keras.layers.Dense(10, activation = 'softmax')])

# Print network
model.summary()

# Compile network: define optimizer, loss and display metrics
model.compile( optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Train
t1 = time.monotonic()
model.fit(train_img, train_lbl, epochs = 27)
t2 = time.monotonic()
print("Training time: {}".format(t2-t1))

# Evaluate
print("Test set results")
test_params = model.evaluate(test_img, test_lbl)
print(test_params)

# ******************************************************** #
#      Visualize output for each layer                     #
# ******************************************************** #
"""

# extract outputs for each layer
layer_outputs = [layer.output for layer in model.layers]
print("Layer Outputs: len {}".format(len(layer_outputs)))
for i, output in enumerate(layer_outputs):
	print("Output {} type {}".format(i,output.shape))

activation_model = models.Model(inputs = model.input, outputs = layer_outputs)

imgs = [0,1,2,3]
conv = [0,1,2,3]

fig1 = plt.figure(figsize=(20, 10))
fig1.suptitle(" Convolutional and Max Pooling Networks Output")


# make one figure for each clothe type
for clothe_type in np.unique(test_lbl):
	print(" Clothe Type {}".format(clothe_type))
	# find all indices of the current clothe type
	indices = np.where( test_lbl == clothe_type)

	# pick the first n images
	n = 5
	imgs = indices[0][:n]
	axarr = fig1.subplots(n,8)

	for i, img_no in enumerate(imgs):
		print("   Img {}".format(img_no))

		output = activation_model.predict(test_img[img_no].reshape(1,28,28,1))
		output.append(test_img[img_no].reshape(1,28,28,1))

		for layer in range(8):		
			print("      Layer {}".format(layer))
			if (layer < 4):
				axarr[i,layer].imshow(output[layer][0,:,:,0], cmap='viridis')
				axarr[i,layer].set_title( ("MaxPool" if layer%2 else "Conv ")+ " , layer# " + str(layer) )
			elif ( (layer>3) and (layer <7)):
				#print("Hola")
				axarr[i,layer].plot(output[layer][0])
				#print("Len {}".format(len(output[x])))
			else:
				axarr[i,layer].imshow(output[layer][0,:,:,0], cmap='viridis')
				axarr[i,layer].set_title("Original Image")

			axarr[i,layer].grid(False)

		fig1.savefig("Layer Output for Clothe Type {}".format(clothe_type))

"""