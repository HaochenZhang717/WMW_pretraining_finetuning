# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers as tfkl
# # from tensorflow.keras import initializers as tfki
# # from tensorflow_probability import distributions as tfd
# # import tensorflow.keras.mixed_precision as prec
#
#
# layer = tfkl.MultiHeadAttention(num_heads=8, key_dim=64, output_shape=600)
#
# q = tf.random.normal((49, 50, 600))
# k = tf.random.normal((49, 50, 600))
# v = tf.random.normal((49, 50, 600))
#
# output_tensor = layer(q, v, k)
# print(output_tensor.shape)
#
#
# import torch
# p1 = torch.tensor([[1,2,3],[1,2,3]], dtype=torch.float32)
# p1_mask = torch.tensor([-1, 0, -1])
# p1.masked_fill_(p1_mask[None,:].bool(), -float('inf')) # [5, 2, 5]
# print(p1)
#
# import tensorflow as tf
# p1 = tf.constant([[1,2,3],[1,2,3]], dtype=tf.float32)
# p1_mask = tf.constant([-1, 0, -1])
# p1 = tf.where(tf.cast(p1_mask, dtype=bool), -float('inf'), p1)
# print(p1)

import common

