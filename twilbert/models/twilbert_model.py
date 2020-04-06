from twilbert.layers.encoders import (SentenceEncoderBlock,
                                      SentenceEncoderMemoryBlock)
from twilbert.layers.layer_norm import LayerNormalization
from keras.models import Model
from twilbert.activations import Gelu
from keras.initializers import TruncatedNormal as tn
from twilbert.pretraining_losses import masked_loss
from keras import backend as K
from keras.layers import (Input, Add, Dense, Embedding, SpatialDropout1D,
                          TimeDistributed, Lambda)
from twilbert.pretraining_metrics import acc_r, acc_m
from twilbert.optimization.optimizers import ADAM, LAMB
import tensorflow as tf

K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
if type(tf.contrib) != type(tf):
    tf.contrib._warning = None


class TWilBertModel:

    def __init__(self, max_len, vocab_size,
                 embedding_size, encoder_size,
                 attention_size, n_heads,
                 cross_sharing, factorize_embeddings,
                 input_dropout, output_dropout,
                 rop_n_hidden, rop_hidden_size,
                 optimizer, accum_iters,
                 pkm, pkm_params, init_range=0.02,
                 gpu=False, multi_gpu=False, n_gpus=0,
                 input_length=None):

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoder_size = encoder_size
        self.attention_size = attention_size
        self.n_heads = n_heads
        self.cross_sharing = cross_sharing
        self.factorize_embeddings = factorize_embeddings
        self.rop_n_hidden = rop_n_hidden
        self.rop_hidden_size = rop_hidden_size
        self.n_encoders = len(self.encoder_size)
        self.output_dropout = output_dropout
        self.input_dropout = input_dropout
        self.optimizer = optimizer
        self.accum_iters = accum_iters
        self.pkm = pkm
        self.pkm_params = pkm_params
        self.init_range = init_range
        self.gpu = gpu
        self.multi_gpu = multi_gpu
        self.n_gpus = [i for i in range(n_gpus)]
        self.device_str = self.__compose_device()
        self.input_length = input_length

    def __compose_device(self):
        device_str = "/device:"
        if not self.gpu:
            device_str += "CPU:0"
        else:
            device_str += "GPU:"
        return device_str

    def build(self):

        with tf.device("/device:GPU:0"):
            input_tokens = Input(shape=(None,))
            input_positions = Input(shape=(None,))
            input_segments = Input(shape=(None,))

            token_embedding_matrix = Embedding(self.vocab_size + 1,
                                               self.embedding_size,
                                               input_length=self.input_length,
                                               embeddings_initializer=tn(
                                                   stddev=self.init_range))

            pos_embedding_matrix = Embedding((2 * self.max_len) + 4,
                                             self.embedding_size,
                                             input_length=self.input_length,
                                             embeddings_initializer=tn(
                                                 stddev=self.init_range))

            seg_embedding_matrix = Embedding(2, self.embedding_size,
                                             input_length=self.input_length,
                                             embeddings_initializer=tn(
                                                 stddev=self.init_range))

            token_embeddings = token_embedding_matrix(input_tokens)
            position_embeddings = pos_embedding_matrix(input_positions)
            segment_embeddings = seg_embedding_matrix(input_segments)

            sum_embeddings = Add()([token_embeddings,
                                    position_embeddings])
            sum_embeddings = Add()([sum_embeddings,
                                    segment_embeddings])

            if self.factorize_embeddings:
                sum_embeddings = Dense(self.encoder_size[0],
                                       kernel_initializer=tn(
                                           stddev=self.init_range))(
                    sum_embeddings)
                sum_embeddings = Gelu()(sum_embeddings)

            if self.input_dropout != 0.:
                sum_embeddings = SpatialDropout1D(self.input_dropout)(
                    sum_embeddings)

            ant_layer = sum_embeddings

            encoders = []

            if self.cross_sharing:
                first_encoder = SentenceEncoderBlock(self.encoder_size[0],
                                                     self.attention_size[0],
                                                     self.n_heads[0],
                                                     dropout=self.output_dropout,
                                                     init_range=self.init_range)

        flag_mem = 0
        for i in range(self.n_encoders):

            if self.pkm and i in self.pkm_params["in_layers"]:
                encoders.append(
                    SentenceEncoderMemoryBlock(self.encoder_size[0],
                                               self.attention_size[0],
                                               self.n_heads[0],
                                               self.pkm_params,
                                               dropout=self.output_dropout,
                                               init_range=self.init_range))
                flag_mem = 1
            else:
                if self.cross_sharing:
                    encoders.append(first_encoder)
                else:
                    encoders.append(SentenceEncoderBlock(self.encoder_size[0],
                                                         self.attention_size[0],
                                                         self.n_heads[0],
                                                         dropout=self.output_dropout,
                                                         init_range=self.init_range))

            if flag_mem == 1:
                with tf.device("/device:GPU:1"):
                    encoded = encoders[-1](ant_layer)
                    ant_layer = encoded
                flag_mem = 0
                #print("Layer: %d -> %s : Allocated in GPU: %d" % (
                #    i, encoders[-1], 1))
            else:
                with tf.device("/device:GPU:%d" % (i % 2)):
                    encoded = encoders[-1](ant_layer)
                    ant_layer = encoded
                #print("Layer: %d -> %s : Allocated in GPU: %d" % (
                #    i, encoders[-1], (i % 2)))

        # Reply Order Prediction #
        cls_output = Lambda(lambda x: x[:, 0, :])(ant_layer)
        rop_hidden = cls_output
        for i in range(self.rop_n_hidden):
            rop_hidden = Dense(self.rop_hidden_size,
                               kernel_initializer=tn(
                                   self.init_range))(rop_hidden)
            rop_hidden = Gelu()(rop_hidden)
            rop_hidden = LayerNormalization()(rop_hidden)

        output_reply_tweet = Dense(2,
                                   activation="softmax",
                                   kernel_initializer=tn(
                                       self.init_range),
                                   name="rop")(rop_hidden)

        mlm_outputs = TimeDistributed(Dense(self.vocab_size,
                                            activation="softmax",
                                            kernel_initializer=tn(
                                                self.init_range)),
                                      name="mlm")(ant_layer)

        self.model = Model(inputs=[input_tokens,
                                   input_positions,
                                   input_segments],
                           outputs=[output_reply_tweet,
                                    mlm_outputs])

        self.pretrained_model = Model(inputs=[input_tokens,
                                              input_positions,
                                              input_segments],
                                      outputs=ant_layer)

    def compile(self, model):
        if self.optimizer is None:
            opt = "adam"
        else:
            name_opt = self.optimizer.strip().lower()
            if name_opt == "lamb":
                opt = LAMB(accum_iters=self.accum_iters)
            elif name_opt == "adam":
                opt = ADAM(accum_iters=self.accum_iters)
            else:
                opt = "adam"

        model.compile(optimizer=opt,
                      loss={"mlm": masked_loss,
                            "rop": "sparse_categorical_crossentropy"},
                      metrics={"rop": acc_r,
                               "mlm": acc_m})

    def save(self, model, f_name):
        model.save_weights(f_name)

    def load(self, model, f_name):
        model.load_weights(f_name)
