from scale_dot import DotProductAttention
from tensorflow import math
from keras.backend import softmax
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer,Dense

# Implementing the Multihead attention
class MultiHeadAttention(Layer):
    def __init__(self,h,d_k,d_v,d_model,**kwargs):
        super(MultiHeadAttention,self).__init__(**kwargs)
        # Scaled dot product attention
        self.attention = DotProductAttention()
        # Number of attention head to use
        self.heads = h 
        # Dimensionality of the linearly projected queries and keys
        self.d_k = d_k 
        # Dimensionality of the linearly projected values
        self.d_v = d_v
        # Dimensionality of the model
        self.d_model = d_model
        # Learned projection matrix for the queries
        self.W_q = Dense(d_k)
        # Learned projection matrix for the keys
        self.W_k = Dense(d_k)
        # Learned projection matrix for the values
        self.W_v = Dense(d_v)
        # Learned projection matrix for the multi-head output
        self.W_o = Dense(d_model)

    def reshape_tensor(self,x,heads,flag):
        if flag:
            # input tensor: (64,5,64)
            # queries = np.random.random((batch_size, input_seq_length, d_k))
            # Tensor shape after reshaping and transposing:
            # (batch_size,heads,seq_length, -1)
            # Reshaped into: (64,5,8,-1)
            x = tf.reshape(x,shape=(tf.shape(x)[0],tf.shape(x)[1],heads,-1))
            # Transposed into: (64,8,5,-1)
            x = tf.transpose(x, perm=(0,2,1,3))
        else:
            # Reverting the reshaping and transposing operations:
            # (batch_size,seq_length,d_k)
            # from (64,8,5,-1) Transposed into: (64,5,8,-1)
            x = tf.transpose(x, perm=(0,2,1,3))
            # Back into (64,5,64)
            x = tf.reshape(x,shape=(tf.shape(x)[0],tf.shape(x)[1],self.d_k))
        return x
    
    def call(self,queries, keys,values,mask=None):
        # Rearange the queries to be able to compute all heads in parallel
        # input queries: (64,5,64)
        q_reshaped = self.reshape_tensor(self.W_q(queries),self.heads,True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys),self.heads,True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
        
        # Rearange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values),self.heads,True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Compute the multi-head attention output 
        # using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped,
                                        k_reshaped,
                                        v_reshaped,
                                        self.d_k,
                                        mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
        
        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped,self.heads,False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to 
        # the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size,input_seq_length, d_model)
        return self.W_o(output)