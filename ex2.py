import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

# data
# house = 50k
# +1 bedroom = 50k

cost_house = 50000
cost_room = 50000

# no of bedrooms
br = np.arange(0,10,1)
# prices
price = br*cost_room
# normalize prices
price_norm = price/max(price)

#train
model.fit(br, price_norm, epochs = 500)

#test
des_rooms = 20
print("Prediction is: {}, {}".format(model.predict([des_rooms]), type(model.predict([des_rooms]))))
print("Predicted House cost is: {}".format(model.predict([des_rooms])*max(price) + cost_house))
print("True value is {}".format(des_rooms*cost_room+cost_house))