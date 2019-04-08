import os, signal
import time
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import zipfile



import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import models

"""
zip_ref = zipfile.ZipFile("/tmp/happy-or-sad.zip", 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()
"""

class cnnCallback(keras.callbacks.Callback):

    # on epoch end
    def on_epoch_end(self, epoch, logs={}):
        """
        Finish when training accuracy > 0.998
        """ 
        if(logs.get('acc') > 0.999):
            print("\n +----------------------------------------------+ \n"+
                  " |   Accuracy {:.3f} > 0.998. Finishing training  |".format(logs.get('acc'))+
                  "\n +----------------------------------------------+ \n")
            self.model.stop_training = True
       

# ******************************************************** #
#           IMPORT ALL DATASETS                            #
# ******************************************************** #

# Create paths
dataset_dir = "happy_sad_dataset/"
train_dir = dataset_dir + "train/"
val_dir = dataset_dir + "val/"
train_dir_happy = os.path.join(train_dir+"happy/")
train_dir_sad = os.path.join(train_dir+"sad/")

# Create Dataset Generators
train_gen = ImageDataGenerator(rescale = 1.0/255.0)
val_gen = ImageDataGenerator(rescale = 1.0/255.0)

# Create "batch iterator" for subsets
train_flow = train_gen.flow_from_directory(
                train_dir,
                target_size = (150, 150),
                batch_size = 4,
                class_mode = 'binary')

"""
val_flow = val_gen.flow_from_directory(
                val_dir,
                target_size = (150, 150),
                batch_size = 4,
                class_mode = 'binary')
"""


 # Define NN model 
convs = 1
model = keras.models.Sequential([
            # 1st Conv Layer
            keras.layers.Conv2D(convs, (3,3), activation = "relu", input_shape= (150,150,3)),
            keras.layers.MaxPooling2D(2,2),
            # 2nd Conv Layer
            keras.layers.Conv2D(convs, (3,3), activation = "relu"),
            keras.layers.MaxPooling2D(2,2),
            # 3rd Conv Layer
            keras.layers.Conv2D(convs, (3,3), activation = "relu"),
            keras.layers.MaxPooling2D(2,2),
            # Flat and Dense Layers
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation = "relu"),
            keras.layers.Dense(1, activation = "sigmoid")
            ])

model.summary()

# Compile network
model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr=0.001), metrics= ['acc'])

# Train network
t1 = time.time()
history = model.fit_generator(train_flow, steps_per_epoch = 100, epochs = 10, verbose = 1, callbacks = [cnnCallback()])
t2 = time.time()
train_time = t2 - t1
print("Training time: {}s".format(train_time))
print("")

# ******************************************************** #
#      Visualize output for each layer                     #
# ******************************************************** #

# Extract outputs for each layer
layer_outputs = [layer.output for layer in model.layers]
layer_names = [layer.name for layer in model.layers]
layer_names.append("Original Image")

activation_model = models.Model(inputs = model.input, outputs = layer_outputs)

# Let's prepare a random input image from the training set.
happy_img_files = [os.path.join(train_dir_happy, happy_image) for happy_image in os.listdir(train_dir_happy)]
sad_img_files = [os.path.join(train_dir_sad, sad_img) for sad_img in os.listdir(train_dir_sad)]


no_images = 5

classes = [happy_img_files[:no_images],sad_img_files[:no_images]]


fig1 = plt.figure(figsize=(20, 10))
fig1.suptitle(" Convolutional and Max Pooling Networks Output")

# draw layer outputs for one sample of each class

print(classes)

for class_image_paths in classes:   

    class_image_number = len(class_image_paths)
    #print(class_image_paths)
    #print("Number of paths {}".format(class_image_number))

    # create array of plots to draw
    axarr = fig1.subplots(class_image_number,len(layer_outputs)+1)  # draw all layers + original image
    #print("axarr shape {}".format(axarr.shape))

    for img_number, img_path in enumerate(class_image_paths):

        # load image
        img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
        x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
        x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

        # get prediction this img
        layers_output = activation_model.predict(x)
        layers_output.append(x)

        for layer_number, output in enumerate(layers_output):
            #print("Output shape: {}".format(output.shape))
            # verify if is convolutional 
            if( len(output.shape) == 4):
                # draw image of Conv 0
                axarr[img_number,layer_number].imshow(output[0,:,:,0], cmap = 'inferno')
                
            else: # or if dense layer
                # draw line
                axarr[img_number, layer_number].plot(output[0])

            # set title
            axarr[img_number,layer_number].set_title(layer_names[layer_number], fontsize='small')
            axarr[img_number,layer_number].tick_params(which='both', bottom = False, left=False, labelbottom=False, labelleft=False)

    fig1.savefig("Layer_Outputs_{}".format(img_path.split('/')[2]), dpi=200)

#plt.show()


# Kill process on exit
print("\nKilling Process... bye!")
os.kill(os.getpid(), signal.SIGKILL)