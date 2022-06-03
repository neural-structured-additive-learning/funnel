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
    return(
        tf.reduce_sum(tf.multiply(W, tf.matmul(P1, W))) + 
        tf.reduce_sum(tf.multiply(tf.matmul(W, P2), W))
        )

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
            initializer="random_normal",
            regularizer=LinearArrayPenalty(self.P1, self.P2),
            trainable=True,
            dtype="float32"
        )

    def call(self, inputs):
        return tf.matmul(tf.matmul(tf.matmul(tf.cast(inputs, dtype="float32"), self.B1), self.w), self.B2, transpose_b = True)
        

