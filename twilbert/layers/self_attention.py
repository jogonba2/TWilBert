from keras import backend as K
from keras.layers import Layer
from numpy import sqrt
from keras.initializers import TruncatedNormal as tn


class SelfAttention(Layer):

    def __init__(self, d, init_range=0.02, **kwargs):
        self.d = d
        self.init_range = init_range
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.wq = self.add_weight(shape=(input_shape[-1], self.d),
                                  name="Wq",
                                  initializer=tn(
                                      stddev=self.init_range),
                                  trainable=True)

        self.wk = self.add_weight(shape=(input_shape[-1], self.d),
                                  name="Wk",
                                  initializer=tn(
                                      stddev=self.init_range),
                                  trainable=True)

        self.wv = self.add_weight(shape=(input_shape[-1], self.d),
                                  name="Wv",
                                  initializer=tn(
                                      stddev=self.init_range),
                                  trainable=True)

        super(SelfAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        q = K.dot(x, self.wq)
        k = K.dot(x, self.wk)
        v = K.dot(x, self.wv)
        attn = K.softmax(K.batch_dot(q, K.permute_dimensions(k, (0, 2, 1))) /
                         sqrt(self.d))
        return K.batch_dot(attn, v)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d)
