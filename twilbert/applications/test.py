import warnings
import os
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from twilbert.utils.generator import PretrainingGenerator
from twilbert.utils.utils import Utils
from twilbert.models.twilbert_model import TWilBertModel
from twilbert.preprocessing.tokenizer import (FullTokenizer,
                                              convert_ids_to_tokens)
from twilbert.utils.pretraining_metrics import acc_m
from keras import backend as K
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

    if pkm:
        pkm_params = config["model"]["pkm_params"]
    else:
        pkm_params = {}

    mlm_type = config["model"]["masked_lm"]["type"]
    mlm_max_span = config["model"]["masked_lm"]["max_span"]
    mlm_budget = config["model"]["masked_lm"]["budget"]
    mlm_probs = [config["model"]["masked_lm"]["probs"]["mask"],
                 config["model"]["masked_lm"]["probs"]["random"],
                 config["model"]["masked_lm"]["probs"]["keep"]]

    rop_n_hidden = config["model"]["rop"]["n_hidden"]
    rop_hidden_size = config["model"]["rop"]["hidden_size"]

    path_load_weights = config["test"]["path_load_weights"]

    encoder_size = [hidden_size for i in range(n_encoders)]
    attention_size = [attention_size for i in range(n_encoders)]
    n_heads = [n_heads for i in range(n_encoders)]


    # Model definition #
    twilbert_model = TWilBertModel(max_len, vocab_size, embedding_size,
                                   encoder_size, attention_size,
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

    print(model.summary())

    # Training params #
    da = PretrainingGenerator(dataset_file, tokenizer,
                              256, mlm_type,
                              mlm_max_span, mlm_budget,
                              mlm_probs, bucket_min_a,
                              bucket_min_b, bucket_max_a,
                              bucket_max_b, bucket_steps)
    c = np.random.randint(0, 255)
    gen = da.generator()
    (batch_x, position_indices, segment_indices), \
        (batch_rop, batch_mlm) = next(gen)

    x = batch_x[c]
    y_rop = batch_rop[c]
    ref = batch_mlm[c].squeeze(axis=-1)
    preds = model.predict([np.array([x]),
                           np.array([position_indices[0]]),
                           np.array([segment_indices[0]])])
    pred_rop = preds[0].argmax()
    pred_mlm = preds[1].argmax(axis=-1)[0]
    print(("\n"*3) + "X\tRef\tPred\n")
    for i in range(len(batch_x[c])):
        print(x[i], "\t", ref[i], "\t", pred_mlm[i])
    print("MLM Accuracy:", K.eval(acc_m(np.array([batch_mlm[c]]), preds[1])))
    print("Truth ROP: %d | Pred ROP: %d" % (y_rop, pred_rop))
