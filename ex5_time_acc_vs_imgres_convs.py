# ex5 from Tensorflow flow from coursera

import os, signal
import zipfile
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import time
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle


import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import models
from tensorflow.keras import backend as K


class cnnCallback(keras.callbacks.Callback):

    # on epoch end
    def on_epoch_end(self, epoch, logs={}):
        """
        Finish when training accuracy > 0.998
        """ 
        if(logs.get('acc') > 0.998):
            print("\n +----------------------------------------------+ \n"+
                  " |   Accuracy {:.3f} > 0.998. Finishing training  |".format(logs.get('acc'))+
                  "\n +----------------------------------------------+ \n")
            self.model.stop_training = True


# extract dataset
"""
zip_ref = zipfile.ZipFile('horse-or-human.zip', 'r')
zip_ref.extractall('horse_human_dataset/')
zip_ref.close()
"""
dataset_dir = 'horse_human_dataset/'
train_dir = dataset_dir+'train/'
val_dir = dataset_dir+'val/'

train_horse_dir = os.path.join(train_dir+'horses/')
train_human_dir = os.path.join(train_dir+'humans/')

# Create Dataset Generators and set parameters
train_datagen = ImageDataGenerator(rescale= 1.0/255.0)
validation_datagen = ImageDataGenerator(rescale = 1.0/255.0)

# Try different input image resolutions
img_res_test = [150, 180, 210, 240, 270]

time_acc = list()

for img_res in img_res_test:


    # Flow train images in batch size = 128
    train_flow = train_datagen.flow_from_directory(
                    train_dir,   # source directory for train images
                    target_size = (img_res, img_res),
                    batch_size = 128,   
                    class_mode = 'binary')

    val_flow = validation_datagen.flow_from_directory( 
                    val_dir,
                    target_size = (img_res, img_res),
                    batch_size = 32,
                    class_mode = 'binary')

    


    # Try different convolutions size, here only define only exponents,
    # such that convs = 2**exponent
    conv_arr = [0,1,2,3,4]
    for conv_l1 in [2**i for i in conv_arr]:

        print("\n +----------------------------------------------+ \n"+
              " |   Resolution {} Convolutions {}  |".format(img_res, conv_l1)+
              "\n +----------------------------------------------+ \n")

        # define neural network model
        model = keras.models.Sequential([
                # First Convolution Layer, input is img_resximg_resx3 for RGB channels
                keras.layers.Conv2D(conv_l1,(3,3), activation='relu', input_shape=(img_res,img_res,3)),
                keras.layers.MaxPooling2D(2,2),
                # Second Conv Layer
                keras.layers.Conv2D(conv_l1,(3,3), activation='relu'),
                keras.layers.MaxPooling2D(2,2),
                # Third Conv Layer
                keras.layers.Conv2D(conv_l1,(3,3), activation='relu'),
                keras.layers.MaxPooling2D(2,2),
                # Fourth Conv Layer
                keras.layers.Conv2D(conv_l1,(3,3), activation='relu'),
                keras.layers.MaxPooling2D(2,2),
                # Fifth Conv Layer
                keras.layers.Conv2D(conv_l1, (3,3), activation='relu'),
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
        #model.summary()

        # Compile network
        model.compile(loss = 'binary_crossentropy', optimizer = RMSprop(lr=0.001), metrics=['acc'])

        # Train network... and measure training time
        t1 = time.time()
        #history = model.fit_generator(train_flow, steps_per_epoch=8, epochs = 2, verbose=1, validation_data = val_flow, validation_steps=8, callbacks=[cnnCallback()])
        train_history = model.fit_generator(train_flow, steps_per_epoch=8, epochs = 30, verbose=1, callbacks=[cnnCallback()])
        t2 = time.time()
        train_time = t2 - t1
        #print("Training time: {}".format(train_time))

        eval_history = model.evaluate_generator(val_flow)
        #print("EVAL HISTORY: {}".format(eval_history))
        #print(model.metrics_names)

        time_acc.append((train_time, eval_history[1])) # store training time and accuracy

        # ******************************************************** #
        #      Test some images                                    #
        # ******************************************************** #

        """
        # Load images for testing
        test_dir = dataset_dir+'test/'
        for image_file in os.listdir(test_dir):

            img = load_img(test_dir+image_file, target_size=(300,300))
            x = img_to_array(img)
            x = np.expand_dims(x, axis = 0)

            images = np.vstack([x])
            classes = model.predict(images, batch_size = 10)
            #print(classes,type(classes))
            if( classes[0] > 0.5 ):
                print("Input image {} is: HUMAN with probability {}".format(image_file, classes[0][0]))
            else:
                print("Input image {} is: HORSE with probability {}".format(image_file, classes[0][0]))
        """

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


        no_images = 5

        classes = [horse_img_files[:no_images],human_img_files[:no_images]]


        fig1 = plt.figure(figsize=(20, 10))
        fig1.suptitle(" Convolutional and Max Pooling Networks Output\n  {} Convolutions, {}x{} Input Image Resolution".format(conv_l1, img_res, img_res))

        # draw layer outputs for one sample of each class
        for class_image_paths in classes:   

            class_image_number = len(class_image_paths)
            #print(class_image_paths)
            #print("Number of paths {}".format(class_image_number))

            # create array of plots to draw
            axarr = fig1.subplots(class_image_number,len(layer_outputs)+1)  # draw all layers + original image
            #print("axarr shape {}".format(axarr.shape))

            for img_number, img_path in enumerate(class_image_paths):

                # load image
                img = load_img(img_path, target_size=(img_res, img_res))  # this is a PIL image
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
                        axarr[img_number,layer_number].imshow(output[0,:,:,0], cmap = 'viridis')
                        
                    else: # or if dense layer
                        # draw line
                        axarr[img_number, layer_number].plot(output[0])

                    # set title
                    axarr[img_number,layer_number].set_title(layer_names[layer_number], fontsize='small')
                    axarr[img_number,layer_number].tick_params(which='both', bottom = False, left=False, labelbottom=False, labelleft=False)

            fig1.savefig("ex5_imgs/Layer_Outputs_{}_Convs{}_Res{}".format(img_path.split('/')[2], conv_l1, img_res), dpi=200)

        # close figures, if not memory consumption goes up
        plt.close('all')

        # delete model to release memory
        del model
        del activation_model
        K.clear_session()

#plt.show()

# Create figure
metrics_plot = plt.figure(figsize=(20,10))
acc_ax = metrics_plot.add_subplot(121, projection="3d")
ttime_ax = metrics_plot.add_subplot(122, projection="3d")

# Create x-y plane to draw
convs, res = np.meshgrid(conv_arr, img_res_test)

# extract metrics
acc_values = np.array([metrics[1] for metrics in time_acc ])
acc_values = acc_values.reshape(convs.shape)

ttime_values = np.array([metrics[0] for metrics in time_acc ])
ttime_values = ttime_values.reshape(convs.shape)

# plot surface
surf1 = acc_ax.plot_surface(convs, res, acc_values, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
surf2 = ttime_ax.plot_surface(convs, res, ttime_values, cmap=cm.viridis,
                       linewidth=0, antialiased=False)

acc_ax.set_xticks(conv_arr)#np.array([2**i for i in conv_arr]))
acc_ax.set_yticks(img_res_test)#np.array([2**i for i in conv_arr]))
acc_ax.set_xticklabels(np.array([2**i for i in conv_arr]))
#acc_ax.set_yticklabels(np.array([2**i for i in conv_arr]))

acc_ax.set_title("Accuracy vs Convolutions per Layer & Input image resolution")
acc_ax.set_xlabel("Convs per Layer")
acc_ax.set_ylabel("Image Resolution")
acc_ax.set_zlabel("Accuracy")

ttime_ax.set_xticks(conv_arr)#np.array([2**i for i in conv_arr]))
ttime_ax.set_yticks(img_res_test)#np.array([2**i for i in conv_arr]))
ttime_ax.set_xticklabels(np.array([2**i for i in conv_arr]))
#ttime_ax.set_yticklabels(np.array([2**i for i in conv_arr]))

ttime_ax.set_title("Training Time vs Convolutions per Layer & Input image resolution")
ttime_ax.set_xlabel("Convs per Layer")
ttime_ax.set_ylabel("Image Resolution")
ttime_ax.set_zlabel("Training Time")

# Add a color bar which maps values to colors.
metrics_plot.colorbar(surf1, shrink=0.5, aspect=5)
metrics_plot.colorbar(surf2, shrink=0.5, aspect=5)

# save figure
metrics_plot.savefig("Accuracy_Time_vs_Convolutions_&_Input_Resolution.png")

pickle.dump(metrics_plot, open('AccTime_VS_ConvRes.pickle', 'wb'))

plt.show()

# Kill process on exit
#print("\nKilling Process... bye!")
#os.kill(os.getpid(), signal.SIGKILL)




