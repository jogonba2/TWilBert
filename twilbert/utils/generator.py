from twilbert.preprocessing.tokenizer import convert_tokens_to_ids
from twilbert.utils.utils import Utils as ut
from keras.utils.np_utils import to_categorical
import numpy as np


class MultipleFinetuningGenerator:

    def __init__(self, tokenizer, n_classes,
                 min_bucket_a=5, min_bucket_b=5,
                 max_bucket_a=45, max_bucket_b=45,
                 bucket_steps=5, preprocessing=True):
        self.min_bucket_a = min_bucket_a
        self.min_bucket_b = min_bucket_b
        self.max_bucket_a = max_bucket_a
        self.max_bucket_b = max_bucket_b
        self.bucket_steps = bucket_steps
        self.buckets = ut.multiple_finetuning_bucketing(self.min_bucket_a,
                                                        self.min_bucket_b,
                                                        self.max_bucket_a,
                                                        self.max_bucket_b,
                                                        self.bucket_steps)
        self.tokenizer = tokenizer
        self.n_classes = n_classes
        self.preprocessing = ut.preprocessing() if preprocessing else None

    def generator(self, ids, x1, x2, y):
        bucked_samples = {}
        n_samples = len(x1)
        for i in range(n_samples):
            x1_i = x1[i]
            x2_i = x2[i]
            if self.preprocessing:
                x1_i = self.preprocessing(x1_i)
                x2_i = self.preprocessing(x2_i)
            x1_tok = ut.tokenize(x1_i, self.tokenizer)
            x2_tok = ut.tokenize(x2_i, self.tokenizer)
            bucked_samples = self.buckets(ids[i], x1_tok, x2_tok, y[i])

        while True:
            for bucket in bucked_samples:
                bucket_size = len(bucked_samples[bucket])
                bucket_1 = bucket[0]
                bucket_2 = bucket[1]

                position_indices = list(range((bucket_1 + bucket_2 + 3)))
                position_indices = np.array([position_indices
                                             for _ in range(bucket_size)],
                                            dtype="int32")

                segment_indices = [0 for _ in range(bucket_1 + 2)] + \
                                  [1 for _ in range(0, bucket_2 + 1, 1)]
                segment_indices = np.array([segment_indices
                                            for _ in range(bucket_size)],
                                           dtype="int32")
                batch_x = np.zeros((bucket_size, bucket_1 + bucket_2 + 3),
                                   dtype="int32")
                batch_y = np.zeros((bucket_size,), dtype="int32")

                for i in range(bucket_size):
                    ids_i, x1_i, x2_i, y_i = bucked_samples[bucket][i]
                    x = ut.prepare_input(x1_i, x2_i)
                    x_ids = convert_tokens_to_ids(self.tokenizer.vocab, x)
                    batch_x[i] = x_ids
                    batch_y[i] = y_i
                p = np.random.permutation(bucket_size)
                batch_x = batch_x[p]
                batch_y = batch_y[p]
                batch_y = to_categorical(batch_y, num_classes=self.n_classes)
                yield ([batch_x, position_indices, segment_indices], batch_y)


class SingleFinetuningGenerator:

    def __init__(self, tokenizer, n_classes,
                 min_bucket=5, max_bucket=45,
                 bucket_steps=5, preprocessing=True,
                 multi_label=False):
        self.min_bucket = min_bucket
        self.max_bucket = max_bucket
        self.bucket_steps = bucket_steps
        self.buckets = ut.single_finetuning_bucketing(self.min_bucket,
                                                      self.max_bucket,
                                                      self.bucket_steps)
        self.tokenizer = tokenizer
        self.n_classes = n_classes
        self.multi_label = multi_label
        self.preprocessing = ut.preprocessing() if preprocessing else None

    def generator(self, ids, x, y):
        bucked_samples = {}
        lx = len(x)
        for i in range(lx):
            x_i = x[i]
            if self.preprocessing:
                x_i = self.preprocessing(x_i)
            x_tok = ut.tokenize(x_i, self.tokenizer)
            bucked_samples = self.buckets(ids[i], x_tok, y[i])

        while True:
            for bucket in bucked_samples:
                bucket_size = len(bucked_samples[bucket])
                position_indices = list(range((bucket + 2)))
                position_indices = np.array([position_indices
                                             for _ in range(bucket_size)],
                                            dtype="int32")
                segment_indices = [0 for _ in range(bucket + 2)]
                segment_indices = np.array([segment_indices
                                            for _ in range(bucket_size)],
                                           dtype="int32")

                batch_x = np.zeros((bucket_size, bucket + 2), dtype="int32")
                if self.multi_label:
                    batch_y = np.zeros((bucket_size, self.n_classes), dtype="int32")
                else:
                    batch_y = np.zeros((bucket_size,), dtype="int32")
                for i in range(bucket_size):
                    ids_i, x_i, y_i = bucked_samples[bucket][i]
                    x_i = ut.prepare_single_input(x_i)
                    x_ids = convert_tokens_to_ids(self.tokenizer.vocab, x_i)
                    batch_x[i] = x_ids
                    batch_y[i] = y_i
                p = np.random.permutation(bucket_size)
                batch_x = batch_x[p]
                batch_y = batch_y[p]
                if not self.multi_label:
                    batch_y = to_categorical(batch_y, num_classes=self.n_classes)
                yield ([batch_x, position_indices, segment_indices], batch_y)


class DataGenerator:

    def __init__(self, dataset_file, tokenizer,
                 batch_size, mlm_type, mlm_max_span,
                 mask_prob, probs_mlm, min_bucket_a,
                 min_bucket_b, max_bucket_a, max_bucket_b,
                 bucket_steps, use_rop):

        self.dataset_file = dataset_file
        self.batch_size = batch_size if batch_size % 2 == 0 else batch_size + 1
        self.min_bucket_a = min_bucket_a
        self.min_bucket_b = min_bucket_b
        self.max_bucket_a = max_bucket_a
        self.max_bucket_b = max_bucket_b
        self.bucket_steps = bucket_steps
        self.buckets = ut.bucketing(self.min_bucket_a, self.min_bucket_b,
                                    self.max_bucket_a, self.max_bucket_b,
                                    self.bucket_steps,
                                    self.batch_size)
        self.mask_prob = mask_prob
        self.probs_mlm = probs_mlm
        self.mlm_type = mlm_type
        self.mlm_max_span = mlm_max_span
        self.tokenizer = tokenizer
        self.vocab_words = list(self.tokenizer.vocab.keys())[5:]
        self.vocab_size = len(self.vocab_words)
        self.use_rop = use_rop


    def generator(self):
        while True:

            fr = open(self.dataset_file, "r", encoding="utf8")
            fr.readline()
            for line in fr.readlines():
                id_, text, id_reply, reply = line.strip().split("\t")
                text, reply = text.strip(), reply.strip()
                text = ut.tokenize(text, self.tokenizer)
                reply = ut.tokenize(reply, self.tokenizer)

                batch = self.buckets(text, reply, y=1)
                res = self.__batching(batch)
                if res is not None:
                    yield res

                batch = self.buckets(reply, text, y=0)
                res = self.__batching(batch)
                if res is not None:
                    yield res

            fr.close()

    def __batching(self, batch):

        if batch is not None:
            x1, x2, y = batch[0], batch[1], batch[2]
            bucket_1, bucket_2 = len(x1[0]), len(x2[0])
            batch_x = np.zeros((self.batch_size, bucket_1 + bucket_2 + 3),
                               dtype="int32")
            batch_rop = np.zeros(self.batch_size, dtype="int32")
            batch_mlm = np.zeros((self.batch_size, bucket_1 + bucket_2 + 3),
                                 dtype="int32")
            position_indices = list(range((bucket_1 + bucket_2 + 3)))
            position_indices = np.array([position_indices
                                         for _ in range(self.batch_size)],
                                        dtype="int32")
            segment_indices = [0 for _ in range(bucket_1 + 2)] + \
                              [1 for _ in range(0, bucket_2 + 1, 1)]

            segment_indices = np.array([segment_indices
                                        for _ in range(self.batch_size)],
                                       dtype="int32")

            for i in range(self.batch_size):
                x = ut.prepare_input(x1[i], x2[i])
                x_ids = convert_tokens_to_ids(self.tokenizer.vocab, x)
                masked_x, mask = None, None
                if self.mlm_type == "token":
                    try:
                        masked_x, mask = ut.mask_tokens(x,
                                                        self.mask_prob,
                                                        self.probs_mlm,
                                                        self.vocab_words)
                    except:
                        print("Error sample")
                        continue

                elif self.mlm_type == "span":
                    try:
                        masked_x, mask = ut.mask_spans(x,
                                                       self.mask_prob,
                                                       self.probs_mlm,
                                                       self.vocab_words,
                                                       self.mlm_max_span)
                    except:
                        print("Error sample")
                        continue

                mask = np.array(mask, dtype="int")
                masked_x_ids = convert_tokens_to_ids(self.tokenizer.vocab,
                                                     masked_x)
                mlm_output = ut.prepare_mlm_output(x_ids, mask)
                batch_x[i] = masked_x_ids
                batch_mlm[i] = mlm_output
                batch_rop[i] = y[i]
            p = np.random.permutation(self.batch_size)
            batch_x = batch_x[p]
            batch_rop = batch_rop[p]
            batch_mlm = batch_mlm[p]
            batch_mlm = np.expand_dims(batch_mlm, -1)
            if self.use_rop:
                return ([batch_x, position_indices, segment_indices],
                        [batch_rop, batch_mlm])
            else:
                return ([batch_x, position_indices, segment_indices],
                        [batch_mlm])

