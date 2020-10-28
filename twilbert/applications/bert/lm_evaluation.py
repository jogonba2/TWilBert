import warnings
import os

#warnings.filterwarnings('ignore', category=FutureWarning)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from twilbert.utils.utils import Utils as ut
from twilbert.models.bert import BertModel
from twilbert.preprocessing.tokenizer import (FullTokenizer,
                                              convert_ids_to_tokens,
                                              convert_tokens_to_ids)
from twilbert.utils.pretraining_metrics import acc_m
from keras import backend as K
from math import log
import numpy as np
import json
import sys


if __name__ == "__main__":
    with open(sys.argv[1], "r") as json_file:
        config = json.load(json_file)

    # Dataset #
    dataset_file = config["dataset"]["test_file"]
    vocab_file = config["dataset"]["vocab_file"]

    # Representation #
    max_len = config["representation"]["max_len_training"]
    max_len_test = config["representation"]["max_len_test"]
    tokenizer = FullTokenizer(vocab_file)
    vocab_size = len(tokenizer.vocab)
    bucket_min_a = config["representation"]["bucket_min_a"]
    bucket_min_b = config["representation"]["bucket_min_b"]
    bucket_max_a = config["representation"]["bucket_max_a"]
    bucket_max_b = config["representation"]["bucket_max_b"]
    bucket_steps = config["representation"]["bucket_steps"]

    # Model #
    factorize_embeddings = config["model"]["factorize_embeddings"]
    cross_sharing = config["model"]["cross_sharing"]
    embedding_size = config["model"]["embedding_size"]
    hidden_size = config["model"]["hidden_size"]
    n_encoders = config["model"]["n_encoders"]
    n_heads = config["model"]["n_heads"]
    attention_size = config["model"]["attention_size"]
    attention_size = hidden_size // n_heads if \
        attention_size is None else \
        attention_size
    input_dropout = config["model"]["input_dropout"]
    output_dropout = config["model"]["output_dropout"]

    pkm = config["model"]["pkm"]
    pkm_params = config["model"]["pkm_params"]

    mlm_type = config["model"]["masked_lm"]["type"]
    assert mlm_type in ["token", "span"]
    mlm_max_span = config["model"]["masked_lm"]["max_span"]
    mlm_budget = config["model"]["masked_lm"]["budget"]
    mlm_probs = [config["model"]["masked_lm"]["probs"]["mask"],
                 config["model"]["masked_lm"]["probs"]["random"],
                 config["model"]["masked_lm"]["probs"]["keep"]]

    use_rop = config["model"]["rop"]["use_rop"]
    rop_n_hidden = config["model"]["rop"]["n_hidden"]
    rop_hidden_size = config["model"]["rop"]["hidden_size"]

    # Training #
    path_load_weights = config["test"]["path_load_weights"]

    output_encoder_size = [hidden_size for i in range(n_encoders)]
    attention_size = [attention_size for i in range(n_encoders)]
    n_heads = [n_heads for i in range(n_encoders)]

    # Model definition #
    twilbert_model = BertModel(max_len, vocab_size, embedding_size,
                               output_encoder_size, attention_size,
                               n_heads, cross_sharing,
                               factorize_embeddings,
                               input_dropout, output_dropout,
                               rop_n_hidden, rop_hidden_size,
                               None, None, pkm, pkm_params,
                               input_length=None, use_rop=use_rop)

    twilbert_model.build()
    model = twilbert_model.model

    twilbert_model.compile(model)

    twilbert_model.load(model, path_load_weights)
    print(model.summary())

    dataset = ut.load_lm_dataset(dataset_file)
    preprocess = ut.preprocessing()
    dataset = [ut.tokenize(preprocess(text), tokenizer) for text in dataset]
    gamma = 0.
    N = len(dataset)
    for i in range(N):
        if i % 50 == 0:
            print("T=%d P(X)=%.3f" % (i+1, (gamma / (i+1))))
        X = ut.prepare_single_input(dataset[i]) # Añadir [CLS] y [SEP]
        T = len(X)
        # Cada muestra X tiene tantos posibles enmascaramientos como |X| #
        maskings = [ut.mask_lm_eval(X, t) for t in range(T)][1:-1]
        c = 1
        alpha = 0
        for masking in maskings:
            x = convert_tokens_to_ids(tokenizer.vocab, masking)
            lx = len(x)
            position_indices = list(range(lx))
            segment_indices = [0 for _ in range(lx)]
            if use_rop:
                preds = model.predict([np.array([x]),
                                       np.array([position_indices]),
                                       np.array([segment_indices])])[1][0][c]
            else:
                preds = model.predict([np.array([x]),
                                       np.array([position_indices]),
                                       np.array([segment_indices])])[0][c]
            id_word = tokenizer.vocab[X[c]]
            phi_t = preds[id_word]
            alpha += log(phi_t)
            c += 1
        alpha /= T
        gamma += alpha
    print("P(X)=%.3f" % (gamma / N))
