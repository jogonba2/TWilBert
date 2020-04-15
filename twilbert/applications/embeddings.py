import warnings
import os
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from twilbert.utils.utils import Utils as ut
from twilbert.models.twilbert_model import TWilBertModel
from twilbert.preprocessing.tokenizer import (FullTokenizer,
                                              convert_ids_to_tokens,
                                              convert_tokens_to_ids)
from twilbert.utils.pretraining_metrics import acc_m
from keras import backend as K
from math import log
import numpy as np
import json
import sys
import h5py


if __name__ == "__main__":
    with open(sys.argv[1], "r") as json_file:
        config = json.load(json_file)
    ut.set_random_seed()

    # Dataset #
    dataset_file = config["dataset"]["test_file"]
    output_file = config["dataset"]["output_file"]
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
    if pkm:
        pkm_params = config["model"]["pkm_params"]
    else:
        pkm_params = {}

    rop_n_hidden = config["model"]["rop"]["n_hidden"]
    rop_hidden_size = config["model"]["rop"]["hidden_size"]
    output_encoder_size = [hidden_size for i in range(n_encoders)]
    attention_size = [attention_size for i in range(n_encoders)]
    n_heads = [n_heads for i in range(n_encoders)]
    path_load_weights = config["test"]["path_load_weights"]

    # Model definition #
    twilbert_model = TWilBertModel(max_len, vocab_size, embedding_size,
                                   output_encoder_size, attention_size,
                                   n_heads, cross_sharing,
                                   factorize_embeddings,
                                   input_dropout, output_dropout,
                                   rop_n_hidden, rop_hidden_size,
                                   None, None, pkm, pkm_params,
                                   input_length=None)

    twilbert_model.build()
    model = twilbert_model.model

    twilbert_model.compile(model)

    twilbert_model.load(model, path_load_weights)
    model = twilbert_model.pretrained_model

    print(model.summary())

    fr = open(dataset_file, "r", encoding="utf8")
    tweets = [line.strip() for line in fr.readlines()]
    preprocess = ut.preprocessing()
    tweets = [ut.tokenize(preprocess(text), tokenizer) for text in tweets]
    n_tweets = len(tweets)
    embeddings = []
    for i in range(n_tweets):
        X = ut.prepare_single_input(tweets[i])
        indices = convert_tokens_to_ids(tokenizer.vocab, X)
        position_indices = list(range(len(X)))
        segment_indices = [0 for _ in range(len(tweets[i]) + 2)]
        pred = model.predict([np.array([indices]),
                              np.array([position_indices]),
                              np.array([segment_indices])])[0]
        embeddings.append(pred)

    # Save embeddings #
    with h5py.File(output_file, 'w') as hf:
        grp=hf.create_group('embeddings')
        for i in range(len(embeddings)):
            grp.create_dataset(str(i), data=embeddings[i])


    # Load embeddings #
    embeddings = []
    with h5py.File(output_file, 'r') as hf:
        grp = hf["embeddings"]
        embeddings = [e.value for e in grp.values()]