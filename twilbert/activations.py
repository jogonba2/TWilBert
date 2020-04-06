from keras import backend as K
from keras.layers import Layer
import math


class Gelu(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, **kwargs):
        cdf = 0.5 * (1.0 + K.tanh(
            (math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3)))))
        return x * cdf

    def compute_output_shape(self, input_shape):
        return input_shape
