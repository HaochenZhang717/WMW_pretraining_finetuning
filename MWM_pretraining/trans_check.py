import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
# from tensorflow.keras import initializers as tfki
# from tensorflow_probability import distributions as tfd
# import tensorflow.keras.mixed_precision as prec


layer = tfkl.MultiHeadAttention(num_heads=8, key_dim=64, output_shape=600)
target = tf.keras.Input(shape=[8, 16])
source = tf.keras.Input(shape=[4, 16])
q = tf.random.normal((49, 50, 600))
k = tf.random.normal((49, 50, 600))
v = tf.random.normal((49, 50, 600))

output_tensor = layer(q, v, k)
print(output_tensor.shape)

# print(weights.shape)