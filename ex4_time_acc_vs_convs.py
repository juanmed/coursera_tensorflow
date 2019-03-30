import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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



class cnnCallback(keras.callbacks.Callback):

	# On epoch end
	def on_epoch_end(self, epoch, logs={}):

		# Finish training if acc>0.99
		if(logs.get('acc') > 0.99):
			print("Reached acc > 0.99. Stop training")
			self.model.stop_training = True

# get fashion dataset
fmnist = keras.datasets.fashion_mnist

# get train, tests sets
(train_img, train_lbl), (test_img, test_lbl) = fmnist.load_data()

# Reshape and Normalize
train_img = train_img.reshape(60000, 28, 28, 1)/255.0
test_img = test_img.reshape(10000, 28, 28, 1)/255.0

conv_arr = [0,1,2] #[0,1,2,3,4,5,6,7]

time_acc = list()
for conv_l1 in [2**i for i in conv_arr]:
	for conv_l2 in [2**j for j in conv_arr]:	

		print("Conv L1 {}  Conv L2 {}".format(conv_l1, conv_l2))

		# create network
		model = keras.models.Sequential([
			keras.layers.Conv2D(conv_l1, (3,3), activation='relu', input_shape=(28,28,1)),
			keras.layers.MaxPooling2D(2,2),
			keras.layers.Conv2D(conv_l2, (3,3), activation='relu'),
			keras.layers.MaxPooling2D(2,2),
			keras.layers.Flatten(),
			keras.layers.Dense(128, activation = 'relu'),
			keras.layers.Dense(10, activation = 'softmax')])


		# Compile network: define optimizer, loss and display metrics
		model.compile( optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


		training_interrupt_callback = cnnCallback()

		# Train
		t1 = time.monotonic()
		model.fit(train_img, train_lbl, epochs = 100, callbacks = [training_interrupt_callback])
		t2 = time.monotonic()
		train_time = t2-t1
		print("Training time: {}".format(train_time))

		# Evaluate
		print("Test set results")
		test_params = model.evaluate(test_img, test_lbl)
		print(test_params)

		# store metrics
		time_acc.append((train_time, test_params[1]))


print(time_acc)


# ******************************************************** #
#      Generate graphs 
# ******************************************************** #

metrics_plot = plt.figure(figsize=(20,10))
acc1 = metrics_plot.add_subplot(121, projection='3d')
acc2 = metrics_plot.add_subplot(122, projection='3d')


# generate X-Y grid for 3D Plot
conv_l1 = np.array([2**i for i in conv_arr])
conv_l2 = np.array([2**i for i in conv_arr])
conv_l1, conv_l2 = np.meshgrid(conv_l1, conv_l2)

# extract accuracy values
acc = np.array([metric[1] for metric in time_acc])
acc = acc.reshape(conv_l1.shape)

# extract training time values
ttime = np.array([metric[0] for metric in time_acc])
ttime = ttime.reshape(conv_l1.shape)

# plot surface
surf1 = acc1.plot_surface(conv_l1, conv_l2, acc, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
surf2 = acc2.plot_surface(conv_l1, conv_l2, ttime, cmap=cm.viridis,
                       linewidth=0, antialiased=False)

acc1.set_xticks(np.array([2**i for i in conv_arr]))
acc1.set_yticks(np.array([2**i for i in conv_arr]))
acc1.set_title("Accuracy vs Convolutions in each Layer")
acc1.set_xlabel("Convs Layer 1")
acc1.set_ylabel("Convs Layer 2")
acc1.set_zlabel("Accuracy")
acc2.set_xticks(np.array([2**i for i in conv_arr]))
acc2.set_yticks(np.array([2**i for i in conv_arr]))
acc2.set_title("Training Time vs Convolutions in each Layer")
acc2.set_xlabel("Convs Layer 1")
acc2.set_ylabel("Convs Layer 2")
acc2.set_zlabel("Training Time")

# Add a color bar which maps values to colors.
metrics_plot.colorbar(surf1, shrink=0.5, aspect=5)
metrics_plot.colorbar(surf2, shrink=0.5, aspect=5)

# save figure
metrics_plot.savefig("Accuracy_Time_vs_Convolutions.png")
