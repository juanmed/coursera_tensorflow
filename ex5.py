# ex5 from Tensorflow flow from coursera

import os 
import zipfile
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import time
import numpy as np
import random


import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import models


class cnnCallback(keras.callbacks.Callback):

    # on epoch end
    def on_epoch_end(self, epoch, logs={}):
        """
        Finish when training accuracy > 0.99
        """ 
        if(logs.get('acc') > 0.99):
            print("\n +----------------------------------------------+ \n"+
                  " |   Accuracy {:.3f} > 0.99. Finishing training  |".format(logs.get('acc'))+
                  "\n +----------------------------------------------+ \n")
            self.model.stop_training = True


# extract dataset
"""
zip_ref = zipfile.ZipFile('horse-or-human.zip', 'r')
zip_ref.extractall('horse_human_dataset/')
zip_ref.close()
"""
dataset_dir = 'horse_human_dataset/'

train_horse_dir = os.path.join(dataset_dir+'horses/')
train_human_dir = os.path.join(dataset_dir+'humans/')

# Create Dataset Generators and set parameters
train_datagen = ImageDataGenerator(rescale= 1.0/255.0)

# Flow train images in batch size = 128
train_flow = train_datagen.flow_from_directory(
                dataset_dir,   # source directory for train images
                target_size=(300, 300),
                batch_size = 128,   
                class_mode='binary')

# define neural network model
model = keras.models.Sequential([
        # First Convolution Layer, input is 300x300x3 for RGB channels
        keras.layers.Conv2D(16,(3,3), activation='relu', input_shape=(300,300,3)),
        keras.layers.MaxPooling2D(2,2),
        # Second Conv Layer
        keras.layers.Conv2D(32,(3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # Third Conv Layer
        keras.layers.Conv2D(64,(3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # Fourth Conv Layer
        keras.layers.Conv2D(64,(3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # Fifth Conv Layer
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # Flatten input
        keras.layers.Flatten(),
        # Dense Hidden Layer
        keras.layers.Dense(512, activation='relu'),
        # Finally, 1 neuron layer!... all reduces to a single output
        # Remember this is binary classification
        keras.layers.Dense(1, activation='sigmoid')
        ])

# Print network structure and outputs
model.summary()

# Compile network
model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr=0.001), metrics=['acc'])

# Train network... and measure training time
t1 = time.time()
history = model.fit_generator(train_flow, steps_per_epoch=8, epochs = 15, verbose=1, callbacks=[cnnCallback()])
t2 = time.time()
print("Training time: {}".format(t2 - t1))

# Load images for testing
testset_dir = dataset_dir+'test/'
for image_file in os.listdir(testset_dir):

    img = load_img(testset_dir+image_file, target_size=(300,300))
    x = img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size = 10)
    #print(classes,type(classes))
    if( classes[0] > 0.5 ):
        print("Input image {} is: HUMAN with probability {}".format(image_file, classes[0][0]))
    else:
        print("Input image {} is: HORSE with probability {}".format(image_file, classes[0][0]))

# ******************************************************** #
#      Visualize output for each layer                     #
# ******************************************************** #

# Extract outputs for each layer
layer_outputs = [layer.output for layer in model.layers]
layer_names = [layer.name for layer in model.layers]
layer_names.append("Original Image")

activation_model = models.Model(inputs = model.input, outputs = layer_outputs)

# Let's prepare a random input image from the training set.
horse_img_files = [os.path.join(train_horse_dir, horse_image) for horse_image in os.listdir(train_horse_dir)]
human_img_files = [os.path.join(train_human_dir, human_img) for human_img in os.listdir(train_human_dir)]
classes = [[random.choice(horse_img_files),random.choice(horse_img_files)],[random.choice(human_img_files), random.choice(human_img_files)]]


fig1 = plt.figure(figsize=(20, 10))
fig1.suptitle(" Convolutional and Max Pooling Networks Output")

# draw layer outputs for one sample of each class

print(classes)

for class_image_paths in classes:   

    class_image_number = len(class_image_paths)
    print(class_image_paths)
    print("Number of paths {}".format(class_image_number))

    # create array of plots to draw
    axarr = fig1.subplots(class_image_number,len(layer_outputs)+1)  # draw all layers + original image
    print("axarr shape {}".format(axarr.shape))

    for img_number, img_path in enumerate(class_image_paths):

        # load image
        img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
        x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
        x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

        # get prediction this img
        layers_output = activation_model.predict(x)
        layers_output.append(x)

        for layer_number, output in enumerate(layers_output):
            print("Output shape: {}".format(output.shape))
            # verify if is convolutional 
            if( len(output.shape) == 4):
                # draw image
                axarr[img_number,layer_number].imshow(output[0,:,:,0], cmap = 'viridis')
                
            else: # or if dense layer
                # draw line
                axarr[img_number, layer_number].plot(output[0])

            # set title
            axarr[img_number,layer_number].set_title(layer_names[layer_number])

    fig1.savefig("Layer_Outputs_{}".format(img_path.split('/')[1]))

plt.show()







