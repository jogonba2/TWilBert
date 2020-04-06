import re
import html
import math
import csv
import numpy as np
import pandas as pd


class Utils:

    @staticmethod
    def load_multiple_dataset(path, id_header="ID",
                              text_header="TEXT", aux_header="AUX",
                              class_header="CLASS", categories=None,
                              delimiter="\t"):

        f = pd.read_csv(path, delimiter=delimiter,
                        encoding="utf8", quoting=csv.QUOTE_NONE)
        ids = f[id_header].tolist()
        texts = f[text_header].tolist()
        aux = f[aux_header].tolist()
        classes = f[class_header].tolist()
        del f
        if categories:
            classes = [categories[str(k)] for k in classes]
        return ids, texts, aux, classes

    @staticmethod
    def load_dataset(path, id_header="ID", text_header="TEXT",
                     class_header="CLASS", categories=None,
                     multi_label=False, delimiter="\t"):

        f = pd.read_csv(path, delimiter=delimiter,
                        encoding="utf8", quoting=csv.QUOTE_NONE)
        ids = f[id_header].tolist()
        texts = f[text_header].tolist()
        classes = f[class_header].tolist()
        del f
        if multi_label:
            classes = [c.split("|") for c in classes]
        if categories:
            if multi_label:
                classes = [[categories[str(k)] for k in c] for c in classes]
            else:
                classes = [categories[str(k)] for k in classes]
        return ids, texts, classes

    @staticmethod
    def load_lm_dataset(path):
        fr = open(path, encoding="utf8")
        dataset = [line.strip() for line in fr.readlines()]
        fr.close()
        return dataset

    """Truncating at end, Padding at start"""
    @staticmethod
    def pad_tweet(tok_tweet, max_len):
        res = tok_tweet[:max_len]
        while len(res) < max_len:
            res.insert(0, "[PAD]")
        return res

    @staticmethod
    def tokenize(tweet, tokenizer):
        return tokenizer.tokenize(tweet)

    @staticmethod
    def full_prepare_tweet(tweet, max_len, tokenizer):
        tweet = tokenizer.tokenize(tweet)
        tweet = Utils.pad_tweet(tweet, max_len)
        return tweet

    @staticmethod
    def prepare_input(tweet, reply):
        return ["[CLS]"] + tweet + ["[SEP]"] + reply + ["[SEP]"]

    @staticmethod
    def prepare_single_input(tweet):
        return ["[CLS]"] + tweet + ["[SEP]"]

    @staticmethod
    def get_segment_indices(x):
        sep_pos = x.index("[SEP]")
        lx = len(x)
        return [0 for _ in range(sep_pos + 1)] + \
               [1 for _ in range(sep_pos + 1, lx)]

    @staticmethod
    def prepare_mlm_output(token_x_indices, mask):
        return token_x_indices * mask

    @staticmethod
    def mask_spans(x, mask_prob, mlm_probs, vocab_words, max_span):
        lx = len(x)
        denom = sum([1. / k for k in range(1, max_span + 1)])
        span_probs = [(1. / n) / denom for n in range(1, max_span + 1)]
        span_lens = [i for i in range(1, max_span + 1)]
        masked_input = x[:]
        masked_flags = np.zeros(lx)
        masked_input = np.array(masked_input, dtype="object")
        valid_pos = np.argwhere((masked_input != "[SEP]") &
                                (masked_input != "[CLS]") &
                                (masked_input != "[PAD]") &
                                (masked_input != "[UNK]")).squeeze(axis=-1)
        n_to_mask = int(math.ceil(mask_prob * len(valid_pos)))
        n_masked = 0

        while n_masked < n_to_mask:
            span_len = np.random.choice(span_lens, p=span_probs)
            zero_pos = np.argwhere(masked_flags == 0).squeeze(axis=-1)
            valid_pos_2 = zero_pos[zero_pos < lx - span_len + 1]
            selected_pos = np.random.choice(np.intersect1d(valid_pos,
                                                           valid_pos_2))
            for i in range(span_len):
                if masked_input[selected_pos + i] not in \
                   ["[SEP]", "[CLS]", "[UNK]", "[MASK]"]:
                    type_mask = np.random.choice([0, 1, 2], p=mlm_probs)
                    if type_mask == 0:
                        masked_input[selected_pos + i] = "[MASK]"
                    elif type_mask == 1:
                        masked_input[selected_pos + i] = np.random.choice(vocab_words)
                    masked_flags[selected_pos + i] = 1
                    n_masked += 1

        return masked_input, masked_flags

    @staticmethod
    def mask_tokens(x, mask_prob, mlm_probs, vocab_words):
        mask, p_mask = [0, 1], [1. - mask_prob, mask_prob]
        type_mask = [0, 1, 2]
        flag_masked = False
        masked_input = []
        masked_flags = []
        while not flag_masked:
            masked_input = []
            masked_flags = []
            for w in x:
                if w in ["[CLS]", "[PAD]", "[SEP]", "[UNK]"]:
                    masked_input.append(w)
                    masked_flags.append(0)

                else:
                    flag_mask = np.random.choice(mask, p=p_mask)
                    if flag_mask == 1:
                        flag_type_mask = np.random.choice(type_mask,
                                                          p=mlm_probs)
                        if flag_type_mask == 0:
                            masked_input.append("[MASK]")
                        elif flag_type_mask == 1:
                            masked_input.append(np.random.choice(vocab_words))
                        else:
                            masked_input.append(w)
                        flag_masked = True
                        masked_flags.append(1)

                    else:
                        masked_input.append(w)
                        masked_flags.append(0)

        return masked_input, masked_flags

    @staticmethod
    def mask_lm_eval(x, t):
        masked_input = []
        lx = len(x)
        for i in range(lx):
            if i>0 and i<lx-1:
                if i==t:
                    masked_input.append("[MASK]")
                else:
                    masked_input.append(x[i])
            else:
                masked_input.append(x[i])
        return masked_input

    @staticmethod
    def max_ratio_weights(labels):
        labels_dict = {}
        for i in range(len(labels)):
            if labels[i] not in labels_dict:
                labels_dict[labels[i]] = 1
            else:
                labels_dict[labels[i]] += 1
        mx = max(labels_dict.items(), key=lambda k: k[1])
        class_weight = {}
        for k in labels_dict:
            class_weight[k] = mx[1] / labels_dict[k]

        return class_weight

    @staticmethod
    def determine_bucket(lx, ranges):
        act_bucket = ranges - lx
        all_neg = (act_bucket < 0)
        zeros = (act_bucket == 0)
        if all_neg.all():
            act_bucket = -1
        elif zeros.any():
            act_bucket = np.argmax(act_bucket > 0)
            act_bucket -= 1
        else:
            act_bucket = np.argmax(act_bucket > 0)
        return ranges[act_bucket]

    @staticmethod
    def bucketing(min_1=5, min_2=5,
                  max_1=40, max_2=40,
                  step=5, max_bucket_size=32):

        ranges_1 = np.array([i for i in range(min_1, max_1, step)])
        ranges_2 = np.array([i for i in range(min_2, max_2, step)])
        buckets = {(i, j): [] for i in ranges_1 for j in ranges_2}
        len_buckets = {(i, j): 0 for i in ranges_1 for j in ranges_2}

        def add(x1, x2, y):
            lx1 = len(x1)
            lx2 = len(x2)
            bucket_1 = Utils.determine_bucket(lx1, ranges_1)
            bucket_2 = Utils.determine_bucket(lx2, ranges_2)
            x1 = Utils.pad_tweet(x1, bucket_1)
            x2 = Utils.pad_tweet(x2, bucket_2)
            buckets[(bucket_1, bucket_2)].append((x1, x2, y))
            len_buckets[(bucket_1, bucket_2)] += 1
            act_bucket = buckets[(bucket_1, bucket_2)]
            if len_buckets[(bucket_1, bucket_2)] == max_bucket_size:
                (x1, x2, y) = zip(*act_bucket)
                buckets[(bucket_1, bucket_2)] = []
                len_buckets[(bucket_1, bucket_2)] = 0
                return (x1, x2, y)

        return add

    @staticmethod
    def single_finetuning_bucketing(min_len, max_len, step):
        ranges = np.array([i for i in range(min_len, max_len, step)])
        buckets = {i: [] for i in ranges}

        def add(id_, x, y):
            lx = len(x)
            bucket = Utils.determine_bucket(lx, ranges)
            x = Utils.pad_tweet(x, bucket)
            buckets[bucket].append((id_, x, y))
            return buckets

        return add

    @staticmethod
    def multiple_finetuning_bucketing(min_1=5, min_2=5,
                                      max_1=40, max_2=40, step=5):

        ranges_1 = np.array([i for i in range(min_1, max_1, step)])
        ranges_2 = np.array([i for i in range(min_2, max_2, step)])
        buckets = {(i, j): [] for i in ranges_1 for j in ranges_2}

        def add(id_, x1, x2, y):
            lx1 = len(x1)
            lx2 = len(x2)
            bucket_1 = Utils.determine_bucket(lx1, ranges_1)
            bucket_2 = Utils.determine_bucket(lx2, ranges_2)
            x1 = Utils.pad_tweet(x1, bucket_1)
            x2 = Utils.pad_tweet(x2, bucket_2)
            buckets[(bucket_1, bucket_2)].append((id_, x1, x2, y))
            return buckets

        return add

    @staticmethod
    def get_best_params(h_res, mon_metric):
        best_params = None
        best_it = None
        best_result = -1
        for params in h_res:
            for it in h_res[params]:
                result = h_res[params][it]
                if result[mon_metric] > best_result:
                    best_params = params
                    best_it = it
                    best_result = result[mon_metric]
        return best_params, best_it

    @staticmethod
    def preprocessing():
        regex_urls = re.compile(r"https?://\S+")
        regex_mentions = re.compile(r"(@+\w+)")

        def preprocess(x):
            r = regex_urls.sub("url", x)
            r = regex_mentions.sub("user", r)
            r = r.replace("…", " ")
            r = r.replace("RT", " ").strip()
            r = html.unescape(r).strip()
            r = " ".join(r.split())
            return r

        return preprocess
