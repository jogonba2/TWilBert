from keras.layers import Layer
from keras.initializers import Ones, Zeros
from keras import backend as K


class LayerNormalization(Layer):
    def __init__(self, eps: float = 1e-12, **kwargs) -> None:
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=input_shape[-1:],
                                     initializer=Ones(),
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[-1:],
                                    initializer=Zeros(),
                                    trainable=True)
        super().build(input_shape)

    def call(self, x, **kwargs):
        u = K.mean(x, axis=-1, keepdims=True)
        s = K.mean(K.square(x - u), axis=-1, keepdims=True)
        z = (x - u) / K.sqrt(s + self.eps)
        return self.gamma * z + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'eps': self.eps,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
