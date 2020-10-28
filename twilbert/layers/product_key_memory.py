from keras.layers import (Layer, Dense, BatchNormalization,
                          Embedding, Dropout)
from keras import backend as K
from keras.initializers import TruncatedNormal as tn
import tensorflow as tf
import numpy as np
import math


class PKM(Layer):

    """
    Adapted from
    https://github.com/facebookresearch/XLM/blob/master/PKM-layer.ipynb
    """

    def __init__(self, k_dim, memory_size,
                 output_dim, n_heads, knn,
                 input_dropout, output_dropout,
                 batch_norm, factorize_embeddings,
                 init_range, **kwargs):

        self.k_dim = k_dim
        self.memory_size = memory_size
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.input_dropout = input_dropout
        self.output_dropout = output_dropout
        self.batch_norm = batch_norm
        self.knn = knn
        self.factorize_embeddings = factorize_embeddings
        self.init_range = init_range
        self.initialize_keys()

        assert self.k_dim >= 2 and self.k_dim % 2 == 0

        super(PKM, self).__init__(**kwargs)

    def build(self, input_shape):
        # Query Layers #
        self.query_layers = []
        for i in range(self.n_heads):
            self.query_layers.append(Dense(self.k_dim, kernel_initializer=tn(
                stddev=self.init_range)))

        for i in range(self.n_heads):
            self.query_layers[i].build(input_shape)
            self._trainable_weights += self.query_layers[i].trainable_weights

        # Value Embeddings #
        self.values = Embedding(self.memory_size ** 2, self.output_dim,
                                embeddings_initializer=tn(
                                   stddev=self.init_range))

        self.values.build(input_shape)

        self._trainable_weights += self.values.trainable_weights

        # Keys #
        self._trainable_weights += [self.keys]

        super(PKM, self).build(input_shape)

    def get_uniform_keys(self):
        half = self.k_dim // 2
        bound = 1 / math.sqrt(half)
        keys = np.random.uniform(-bound, bound, (self.memory_size,
                                                 half))
        return keys

    def initialize_keys(self):

        """
        Create two subkey sets per head.
        `self.keys` is of shape (heads, 2, n_keys, k_dim // 2)
        """

        half = self.k_dim // 2
        nkeys = np.array([self.get_uniform_keys()
                          for _ in range(self.n_heads)
                          for _ in range(2)])

        nkeys = nkeys.reshape((self.n_heads, 2,
                               self.memory_size,
                               half))

        self.keys = K.variable(nkeys)

    def compute_mask(self, inputs, mask=None):
        return mask

    # https://stackoverflow.com/questions/51143210/batched-gather-gathernd #
    @staticmethod
    def __batched_gather(values, indices):
        row_indices = tf.range(0, tf.shape(values)[0])[:, tf.newaxis]
        row_indices = tf.tile(row_indices, [1, tf.shape(indices)[-1]])
        indices = tf.stack([row_indices, indices], axis=-1)
        return tf.gather_nd(values, indices)

    def _get_indices(self, query, subkeys):

        """
        Generate scores and indices for a specific head.
        """

        knn = self.knn
        half = self.k_dim // 2
        n_keys = subkeys[0].shape[0]

        # split query for product quantization
        q1 = query[:, :half]
        q2 = query[:, half:]

        # compute indices with associated scores
        scores1 = K.dot(q1, K.permute_dimensions(subkeys[0], (1, 0)))
        scores2 = K.dot(q2, K.permute_dimensions(subkeys[1], (1, 0)))
        scores1, indices1 = tf.nn.top_k(scores1, k=knn)
        scores2, indices2 = tf.nn.top_k(scores2, k=knn)

        # cartesian product on best candidate keys
        all_scores = K.reshape(K.repeat_elements(K.expand_dims(scores1,
                                                               axis=-1),
                                                 knn, axis=-1) +
                               K.repeat_elements(K.expand_dims(scores2,
                                                               axis=1),
                                                 knn, axis=1),
                               (-1, knn * knn))

        all_indices = K.reshape(K.repeat_elements(K.expand_dims(indices1,
                                                                axis=-1),
                                                  knn, axis=-1) * n_keys +
                                K.repeat_elements(K.expand_dims(indices2,
                                                                axis=1),
                                                  knn, axis=1),
                                (-1, knn * knn))

        scores, best_indices = tf.nn.top_k(all_scores, k=knn)
        indices = PKM.__batched_gather(all_indices, best_indices)

        assert scores.shape[-1] == indices.shape[-1] == knn

        return scores, indices

    def get_indices(self, query):

        """
        Generate scores and indices.
        """

        outputs = [self._get_indices(query[:, i], self.keys[i])
                   for i in range(self.n_heads)]
        scores, indices = [], []
        for i in range(len(outputs)):
            scores.append(outputs[i][0])
            indices.append(outputs[i][1])
        scores = K.stack(scores, axis=1)
        indices = K.stack(indices, axis=1)
        return scores, indices

    def call(self, x, mask=None):

        """
        Read from the memory.
        """

        # Compute Query #
        if self.input_dropout != 0:
            x = Dropout(self.input_dropout)(x)

        if self.batch_norm:
            query_by_head = [K.expand_dims(self.query_layers[i](x), axis=1)
                             for i in range(self.n_heads)]
        else:
            query_by_head = [K.expand_dims(BatchNormalization()(
                self.query_layers[i](x)), axis=1)
                for i in range(self.n_heads)]

        query = K.concatenate(query_by_head, axis=1)

        if self.output_dropout != 0:
            query = Dropout(self.output_dropout)(query)

        # Retrieve Indices and Scores #
        scores, indices = self.get_indices(query)
        scores = K.softmax(scores)

        # merge heads / knn (since we sum heads)
        indices = K.reshape(indices, (-1, self.n_heads * self.knn))
        scores = K.reshape(scores, (-1, self.n_heads * self.knn))

        # weighted sum of values
        embeds = self.values(indices)
        output = K.sum(embeds * scores[:, :, tf.newaxis], axis=1)

        if self.output_dropout != 0:
            output = Dropout(self.output_dropout)(output)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
