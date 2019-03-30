# Course 1, Part 4, Lesson 2 Notebook of Introduction 
# to Deep Learning with TensorFlow course 

# Reference Notebook:
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb#scrollTo=q3KzJyjv3rnA

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import time
print ("Using tf version: {}".format(tf.__version__))

# load fmnist dataset
fmnist = keras.datasets.fashion_mnist

# get train and test datasets
(train_images, train_labels), (test_images, test_labels) = fmnist.load_data()
print("fmnist: type {}".format(type(fmnist)))
print("Train images: len {} type {}".format(len(train_images), type(train_images)))
print("Train labels: len {} type {}".format(len(train_labels), type(train_labels)))
print("Test  images: len {} type {}".format(len(test_images), type(test_images)))
print("Test  labels: len {} type {}".format(len(test_labels), type(test_labels)))

train_sample = 5
print("Image size: {}".format(train_images[train_sample].shape))
print("Label: {}".format(train_labels[train_sample]))
print("Total labels: {}".format(np.unique(train_labels)))
#plt.imshow(train_images[train_sample])
#plt.show()

# normalize set
train_images  = train_images / 255.0    
test_images = test_images / 255.0

loss_acc = list()

r =np.arange(3,5,1)
for epoch in r:
    model = keras.models.Sequential([ keras.layers.Dense(28*28, activation=tf.nn.relu), #keras.layers.Flatten(),
                                      keras.layers.Dense(256, activation=tf.nn.relu),
                                      #keras.layers.Dense(256, activation=tf.nn.relu),
                                      keras.layers.Dense(10, activation=tf.nn.softmax) ])

    # pass optimization parameters
    model.compile( optimizer= tf.train.AdamOptimizer(),
               loss = "sparse_categorical_crossentropy",
               metrics = ['accuracy'])

    # train network
    t1 = time.time()
    model.fit(np.array([image.flatten() for image in train_images]), train_labels, epochs = epoch)
    t2 = time.time()
    print("Training time: {}".format(t2-t1))
    # evaluate 
    print("Evaluation set results: ")
    #t1 = time.time()
    eval_res = model.evaluate(np.array([image.flatten() for image in test_images]), test_labels)
    #t2 = time.time()
    print("Evalutaion time: {}".format(t2-t1))
    print("Eval Res Type {} Len {} val {}".format(type(eval_res), len(eval_res), eval_res))

    loss_acc.append([eval_res[0],eval_res[1],t2-t1])



# predict
classifications = model.predict(np.array([image.flatten() for image in test_images]))

print("Classfications: len: {} \n sample: {}".format(classifications.shape, classifications[0]))

fig = plt.figure(figsize=(20,10))
fig2 = plt.figure(figsize=(20,10))
fig.suptitle("Result for training with various epoch", fontsize='small')
fig2.suptitle("Training time with various epoch", fontsize='small')

ax1 = fig.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)

ax1.plot(r,[res[0] for res in loss_acc], linestyle = '-',color ='r', label = "loss")
ax1.plot(r,[res[1] for res in loss_acc], linestyle = '-',color ='g', label = "acc")
ax2.plot(r,[res[2] for res in loss_acc], linestyle = '-',color ='b', label = "time")
ax1.legend(loc='center right', shadow=True, fontsize='small')
ax1.set_xlabel("epoch {}")
ax1.set_label("metrics {m}")
ax1.set_xticks(r)
fig.savefig("Loss_Acc vs Epoch")

ax2.legend(loc='center right', shadow=True, fontsize='small')
ax2.set_xlabel("epoch {}")
ax2.set_label("time {m}")
ax2.set_xticks(r)
fig2.savefig("Time vs Epoch")

# sgd
# 255 - 
# 120 - 0.8566
# 100 - 0.8541
# 60 - 0.8657
# 30 - 0.8605
# 15 - 0.8562
# 7 - 0.8493
# 3 - 0.1999
# 2 - 0.10
# 1 - 0.10

# adam
# 255 - 0.8794
# 120 - 0.8690
# 100 - 0.8723 
# 60 - 0.8710
# 30 - 0.8690
# 15 - 0.8559
# 7 - 0.8508
# 3 - 0.3841
# 2 - 0.3798
# 1 - 0.1944