import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp 

input_data = tf.keras.layers.Input(shape=(1,)) # 

layer1 = tf.keras.layers.BatchNormalization(1)
out1 = layer1(input_data)

# Get weights only returns a non-empty list after we need the input_data
print("layer1.get_weights() =", layer1.get_weights())

# This is actually the required object for weights.
new_weights = [np.array([1]), np.array([0]),np.array([1]), np.array([0])]

layer1.set_weights(new_weights)

out1 = layer1(input_data)
print("layer1.get_weights() =", layer1.get_weights())

func1 =  tf.keras.backend.function([input_data], [layer1.output])

# The input to the layer.
data = np.array([[5], [2], [4]])
print(data)

# # The output of layer1
layer1_output = func1(data)
print("layer1_output =", layer1_output)