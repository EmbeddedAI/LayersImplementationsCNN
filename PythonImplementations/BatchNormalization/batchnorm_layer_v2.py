import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pprint import pprint

import numpy as np
import tensorflow as tf

epsilon = 0#.000001

data_x = [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]
data_y = [0.169610272, 0.283395813, 0.386358738, 0.470227872, 0.433281294, 0.600267648, 0.73833898, 0.79031502, 0.877464268, 0.843564462, 0.964438917]

bn = tf.keras.layers.BatchNormalization(epsilon=epsilon,beta_initializer='zeros',gamma_initializer='ones',moving_mean_initializer='zeros',moving_variance_initializer='ones')

model = tf.keras.Sequential(bn)
model.compile(optimizer='sgd',loss='mean_squared_error')

pprint(dir(bn))
#model.fit(x=data_x, y=data_y, epochs=500)


print(model.predict(data_x))