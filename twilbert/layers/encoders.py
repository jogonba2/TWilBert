from __future__ import absolute_import
from keras.layers import Layer, TimeDistributed, Dense
from keras import backend as K
from twilbert.utils.activations import Gelu
from twilbert.layers.multihead_attention import MultiHeadAttention
from twilbert.layers.product_key_memory import PKM
from twilbert.layers.layer_norm import LayerNormalization
from keras.initializers import TruncatedNormal as tn


class SentenceEncoderBlock(Layer):

    def __init__(self, output_dim, attention_dim,
                 n_heads, dropout=0.3, init_range=0.02, **kwargs):
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.init_range = init_range
        super(SentenceEncoderBlock, self).__init__(**kwargs)

    def build(self, input_shape):

        self.dense_1 = Dense(4 * self.output_dim,
                             kernel_initializer=tn(
                                 stddev=self.init_range))
        self.dense_1.build(input_shape)
        self._trainable_weights += self.dense_1.trainable_weights

        self.dense_2 = Dense(self.output_dim,
                             kernel_initializer=tn(
                                 stddev=self.init_range))
        self.dense_2.build((input_shape[0],
                            input_shape[1], 4 * self.output_dim))
        self._trainable_weights += self.dense_2.trainable_weights

        # Multi Head Attention #
        self.multihead_attention = MultiHeadAttention(self.attention_dim,
                                                      self.n_heads,
                                                      self.init_range)
        self.multihead_attention.build(input_shape)
        self._trainable_weights += self.multihead_attention.trainable_weights

        # LayerNorm #
        self.layer_normalization_1 = LayerNormalization()
        self.layer_normalization_1.build(input_shape)
        self._trainable_weights += self.layer_normalization_1.trainable_weights

        # LayerNorm #
        self.layer_normalization_2 = LayerNormalization()
        self.layer_normalization_2.build(input_shape)
        self._trainable_weights += self.layer_normalization_2.trainable_weights

        # Gelu #
        self.gelu = Gelu()
        self.gelu.build((input_shape[0], input_shape[1], 4 * self.output_dim))

        super(SentenceEncoderBlock, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer
        return mask

    def call(self, x, mask=None):
        z = self.multihead_attention(x)
        if self.dropout != 0:
            z = K.dropout(z, self.dropout)
        xz = self.layer_normalization_1(x + z)
        h_xz = self.dense_1(xz)
        h_xz = self.gelu(h_xz)
        h_xz = self.dense_2(h_xz)
        if self.dropout != 0:
            h_xz = K.dropout(h_xz, self.dropout)
        h_xz = self.layer_normalization_2(h_xz + xz)
        return h_xz

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


class SentenceEncoderMemoryBlock(Layer):

    def __init__(self, output_dim, attention_dim,
                 n_att_heads, pkm_params, dropout=0.1, init_range=0.02,
                 **kwargs):

        self.output_dim = output_dim
        self.n_att_heads = n_att_heads
        self.k_dim = pkm_params["k_dim"]
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.init_range = init_range
        self.memory_size = pkm_params["memory_size"]
        self.n_pkm_heads = pkm_params["n_heads"]
        self.knn = pkm_params["knn"]
        self.pkm_input_dropout = pkm_params["input_dropout"]
        self.pkm_output_dropout = pkm_params["output_dropout"]
        self.pkm_batch_norm = pkm_params["batch_norm"]
        self.pkm_factorize_embeddings = pkm_params["factorize_embeddings"]
        super(SentenceEncoderMemoryBlock, self).__init__(**kwargs)

    def build(self, input_shape):

        # Multi Head Attention #
        self.multihead_attention = MultiHeadAttention(self.attention_dim,
                                                      self.n_att_heads,
                                                      self.init_range)
        self.multihead_attention.build(input_shape)
        self._trainable_weights += self.multihead_attention.trainable_weights

        # Memory Layer #
        self.memory_layer = PKM(self.k_dim, self.memory_size,
                                self.output_dim, self.n_pkm_heads,
                                self.knn, self.pkm_input_dropout,
                                self.pkm_output_dropout,
                                self.pkm_batch_norm,
                                self.pkm_factorize_embeddings,
                                self.init_range)

        self.memory_layer.build(input_shape)
        self._trainable_weights += self.memory_layer.trainable_weights

        # LayerNorm #
        self.layer_normalization_1 = LayerNormalization()
        self.layer_normalization_1.build(input_shape)
        self._trainable_weights += self.layer_normalization_1.trainable_weights

        # LayerNorm #
        self.layer_normalization_2 = LayerNormalization()
        self.layer_normalization_2.build(input_shape)
        self._trainable_weights += self.layer_normalization_2.trainable_weights

        super(SentenceEncoderMemoryBlock, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer
        return mask

    def call(self, x, mask=None):
        z = self.multihead_attention(x)
        if self.dropout != 0:
            z = K.dropout(z, self.dropout)
        xz = self.layer_normalization_1(x + z)
        h_xz = TimeDistributed(self.memory_layer)(xz)
        if self.dropout:
            h_xz = K.dropout(h_xz, self.dropout)
        h_xz = self.layer_normalization_2(h_xz + xz)
        return h_xz

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)
