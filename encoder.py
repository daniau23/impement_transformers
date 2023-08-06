from tensorflow import math
from keras.backend import softmax
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer,Dense,ReLU,\
    Dropout,LayerNormalization
from pos_encoding import PositionEmbeddingFixedWeights
from multihead import MultiHeadAttention


# Add & Normalise layer
class AddNormalisation(Layer):
    def __init__(self,**kwargs):
        super(AddNormalisation,self).__init__(**kwargs)
        # layer normalisation
        self.layer_norm = LayerNormalization() 
    
    def call(self,x,sublayer_x):
        # The sublayer input and output 
        # need to be of the same shape to be summed
        add = x + sublayer_x

        # Normalisation
        return self.layer_norm(add)


# Feed-Forward Layer
class FeedForward(Layer):
    def __init__(self,d_ff,d_model,**kwargs):
        super(FeedForward,self).__init__(**kwargs)
        self.fc1  = Dense(d_ff)
        self.fc2  = Dense(d_model)
        self.activation = ReLU()

    def call(self,x):
        x_fc1 = self.fc1(x)
        activate = self.activation(x_fc1)
        x_fc2 = self.fc2(activate)

        return x_fc2



# Encoding Layer 
class EncodingLayer(Layer):
    def __init__(self,h,d_k,d_v,d_model,d_ff,rate,**kwargs) -> None:
        super(EncodingLayer,self).__init__(**kwargs)
        # Define the layers
        self.multihead_attention = MultiHeadAttention(h,d_k,d_v,d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalisation()
        self.ff = FeedForward(d_ff,d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalisation()

    def call(self,x,padding_mask,training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x,x,x,padding_mask)
        # Expected output shape  = batch_size, seqence_length, d_model

        # Dropout layer
        multihead_output = self.dropout1(multihead_output,training=training)

        # Add & Norm layer
        addnorm_output = self.add_norm1(x,multihead_output)
        # Expected output shape  = batch_size, seqence_length, d_model

        # Feed Forward layer
        ff_output = self.ff(addnorm_output)
        # Expected output shape  = batch_size, seqence_length, d_model

        # Dropout layer
        ff_output = self.dropout2(ff_output)

        # Add & Norm layer
        addnorm_output2 = self.add_norm1(addnorm_output,ff_output)

        return addnorm_output2        


# Implementing the encoder
class Encoder(Layer):
    def __init__(self,vocab_size,sequence_length,
                    h,d_k,
                    d_v,d_model,
                    d_ff,n,rate,
                    **kwargs):
        super(Encoder,self).__init__(**kwargs)

        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length,
                                vocab_size,d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [
            EncodingLayer(h,d_k,d_v,d_model,d_ff,rate,) for _ in range(n)
        ]

    def call(self,input_sentence,padding_mask,training):
        # Generate positional encoding
        pos_encoding_output = self.pos_encoding(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Dropout layer
        x = self.dropout(pos_encoding_output)
        
        # Pass on the positional encoded values to each encoder layer
        for i,layer in enumerate(self.encoder_layer):
            x = layer(x,padding_mask,training)

        return x