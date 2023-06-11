import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
import tensorflow.keras.regularizers as regularizers

def kroneckersumpen_array(W, P1, P2):
    # Define dim(W) = dim(P1)[1] \times dim(P2)[1]
    # vec(W)^\top (I__2 \otimes P_1) vec(W) = vec(W \circle (P_1 W))^\top 1
    # vec(W)^\top (P_2 \otimes I_1) vec(W) = vec((W P_2) \circle W)^\top 1
    # W = tf.cast(W, dtype="float32")
    if P1 is None:
        return(tf.reduce_sum(tf.multiply(tf.matmul(W, P2), W)))
    else:
        return(
            tf.reduce_sum(tf.multiply(W, tf.matmul(P1, W))) + 
            tf.reduce_sum(tf.multiply(tf.matmul(W, P2), W))
            )

# from deepregression
def vecmatvec(a, B, c=None, sparse_mat = False):
    if c is None:
        c = a
    #return(tf.matmul(tf.transpose(a),tf.linalg.matvec(B, tf.squeeze(c, [1]), a_is_sparse = sparse_mat)))
    return(tf.keras.backend.sum(tf.keras.backend.batch_dot(a, tf.keras.backend.dot(B, c), axes=1)))

# from deepregression          
class squaredPenalty(regularizers.Regularizer):

    def __init__(self, P, strength):
        self.strength = strength
        self.P = P

    def __call__(self, x):
        return self.strength * tf.reduce_sum(vecmatvec(x, tf.cast(self.P, dtype="float32"), sparse_mat = True))

    def get_config(self):
        return {'strength': self.strength, 'P': self.P}

class LinearArrayPenalty(regularizers.Regularizer):

    def __init__(self, P1, P2, strength=1):
        self.strength = strength
        self.P1 = P1
        self.P2 = P2

    def __call__(self, x):
        pen = kroneckersumpen_array(x, self.P1, self.P2)
        return self.strength * pen

    def get_config(self):
        return {'strength': self.strength, 
                'P1': self.P1,
                'P2': self.P2}

class LinearArray(keras.layers.Layer):
    def __init__(self, B1, B2, P1=None, P2=None, **kwargs):
        super(LinearArray, self).__init__(**kwargs)
        self.B1 = tf.cast(B1, dtype="float32")
        self.B2 = tf.cast(B2, dtype="float32")
        self.units = (self.B1.shape[1], self.B2.shape[1])
        self.P1 = tf.cast(P1, dtype="float32")
        self.P2 = tf.cast(P2, dtype="float32")

    def build(self, input_shape):
        self.w = self.add_weight(
            shape = self.units,
            initializer="zeros", #tf.random_normal_initializer(),
            regularizer=LinearArrayPenalty(self.P1, self.P2),
            trainable=True,
            dtype="float32"
        )

    def call(self, input):
        return tf.matmul(tf.matmul(tf.matmul(tf.cast(input, dtype="float32"), self.B1), self.w), self.B2, transpose_b = True)
            
    def get_config(self):
        config = super().get_config()
        config.update({
            "B1": self.B1,
            "B2": self.B2,
            "P1": self.P1,
            "P2": self.P2,
        })
        return config

        
class LinearArraySimple(keras.layers.Layer):
    def __init__(self, units, B2, P1=None, P2=None, **kwargs):
        super(LinearArraySimple, self).__init__(**kwargs)
        self.units = units
        self.B2 = tf.cast(B2, dtype="float32")
        self.P1 = P1
        if self.P1 is not None:
            self.P1 = tf.cast(self.P1, dtype="float32")
        self.P2 = tf.cast(P2, dtype="float32")

    def build(self, input_shape):
        self.w = self.add_weight(
            shape = self.units,
            initializer="zeros", #tf.random_normal_initializer(),
            regularizer=LinearArrayPenalty(self.P1, self.P2),
            trainable=True,
            dtype="float32"
        )

    def call(self, inputs):
        return tf.matmul(tf.matmul(tf.cast(inputs, dtype="float32"), self.w), self.B2, transpose_b = True)
            
    def get_config(self):
        config = super().get_config()
        config.update({
            "B2": self.B2,
            "units": self.units,
            "P1": self.P1,
            "P2": self.P2,
        })
        return config      
      
class SofLayer(keras.layers.Layer):
    def __init__(self, units, B1, P1=None, **kwargs):
        super(SofLayer, self).__init__(**kwargs)
        self.units = units
        self.B1 = tf.cast(B1, dtype="float32")
        self.P1 = P1
        if self.P1 is not None:
            self.P1 = tf.cast(self.P1, dtype="float32")
        
    def build(self, input_shape):
        self.w = self.add_weight(
            shape = self.units,
            initializer="zeros", #tf.random_normal_initializer(),
            regularizer=squaredPenalty(self.P1, 1),
            trainable=True,
            dtype="float32"
        )

    def call(self, inputs):
        return tf.matmul(tf.matmul(tf.cast(inputs, dtype="float32"), self.B1), self.w)
        
            
    def get_config(self):
        config = super().get_config()
        config.update({
            "B1": self.B1,
            "units": self.units,
            "P1": self.P1,
        })
        return config
