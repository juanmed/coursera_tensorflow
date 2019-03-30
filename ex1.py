import tensorflow as tf
import numpy as np 
from tensorflow import keras

# create net: 1 layer with 1 neuron with 1 input
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

# data to fit
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# train
model.fit(xs, ys, epochs=500)

# test
print(model.predict([0.0]))
