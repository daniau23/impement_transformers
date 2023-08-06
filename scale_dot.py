from tensorflow import math
from keras.backend import softmax
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

class DotProductAttention(Layer):
    def __init__(self,**kwargs):
        super(DotProductAttention,self).__init__(**kwargs)

    def call(self,queries,keys,values,d_k,mask=None):
        # Scoring the queries agasinst 
        # the keys after transposing the latter, and scaling
        scores = tf.matmul(queries,keys,transpose_b=True)/\
            math.sqrt(tf.cast(d_k,tf.float32))
        # Apply msk to attention scores
        if mask is not None:
            scores+=1e9 *mask

        # Computing the weights by a softmax
        weights = softmax(scores)

        # Computing the attention by a weightd sum of the value vectors
        return tf.matmul(weights,values)