from keras import backend as K
from keras.layers import Layer
from twilbert.layers.self_attention import SelfAttention
from keras.initializers import TruncatedNormal as tn


class MultiHeadAttention(Layer):

    def __init__(self, d, n_heads, init_range=0.02, **kwargs):
        self.d = d
        self.n_heads = n_heads
        self.heads = {i: None for i in range(self.n_heads)}
        self.init_range = init_range
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):

        self.w = self.add_weight(shape=(self.d * self.n_heads,
                                 input_shape[-1]),
                                 name="w",
                                 initializer=tn(stddev=self.init_range),
                                 trainable=True)

        for i in range(self.n_heads):
            self.heads[i] = SelfAttention(self.d, self.init_range)
            self.heads[i].build(input_shape)
            self._trainable_weights += self.heads[i].trainable_weights

        super(MultiHeadAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer
        return mask

    def call(self, x, mask=None):
        all_heads = [self.heads[i](x) for i in range(self.n_heads)]
        z = K.dot(K.concatenate(all_heads, axis=-1), self.w)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[-1])
