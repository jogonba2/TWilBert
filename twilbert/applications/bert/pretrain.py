import warnings
import os
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from twilbert.utils.generator import DataGenerator
from twilbert.models.bert import BertModel
from twilbert.preprocessing.tokenizer import FullTokenizer
from twilbert.utils.pretraining_callbacks import TimeCheckpoint
from math import floor
from keras.callbacks import ModelCheckpoint
from twilbert.optimization.lr_annealing import Noam
import subprocess
import json
import sys


if __name__ == "__main__":

    with open(sys.argv[1], "r") as json_file:
        config = json.load(json_file)

    # Dataset #
    dataset_file = config["dataset"]["file"]
    vocab_file = config["dataset"]["vocab_file"]

    # Representation #
    max_len = config["representation"]["max_len"]
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
    initializer_range = config["model"]["initializer_range"]
    pkm = config["model"]["pkm"]

    if not pkm:
        pkm_params = None
    else:
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
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    optimizer = config["training"]["optimizer"]
    annealing = config["training"]["noam_annealing"]
    warmup_steps = config["training"]["warmup_steps"]
    accum_iters = config["training"]["accum_iters"]
    gpu = config["training"]["use_gpu"]
    multi_gpu = config["training"]["multi_gpu"]
    n_gpus = config["training"]["n_gpus"]
    verbose = config["training"]["verbose"]
    if use_rop:
        path_save_weights = config["training"]["path_save_weights"] + "/" + \
                            "model_{epoch:02d}-{loss:.4f}-{mlm_acc_m:.4f}.hdf5"
    else:
        path_save_weights = config["training"]["path_save_weights"] + "/" + \
                            "model_{epoch:02d}-{loss:.4f}-{acc_m:.4f}.hdf5"
    path_initial_weights = None
    if "path_initial_weights" in config["training"]:
        path_initial_weights = config["training"]["path_initial_weights"]
    if "initial_batch" in config["training"]:
        initial_batch = config["training"]["initial_batch"]
    else:
        initial_batch = 0

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
                               optimizer, accum_iters,
                               pkm, pkm_params, initializer_range,
                               gpu, multi_gpu, n_gpus, use_rop=use_rop)

    twilbert_model.build()
    model = twilbert_model.model
    twilbert_model.compile(model)

    if path_initial_weights:
        twilbert_model.load(model, path_initial_weights)
    print(model.summary())

    # Training params #
    da = DataGenerator(dataset_file, tokenizer,
                       batch_size, mlm_type,
                       mlm_max_span, mlm_budget,
                       mlm_probs, bucket_min_a,
                       bucket_min_b, bucket_max_a,
                       bucket_max_b, bucket_steps,
                       use_rop=use_rop)

    checkpoint = ModelCheckpoint(path_save_weights,
                                 monitor="loss",
                                 save_best_only=False)

    hour_checkpoint = TimeCheckpoint(hours_step=1,
                                     path=config["training"][
                                         "path_save_weights"])
    callbacks = [checkpoint, hour_checkpoint]

    if annealing:
        noam = Noam(warmup_steps=warmup_steps,
                    hidden_dims=hidden_size,
                    accum_iters=accum_iters,
                    initial_batch=initial_batch)
        callbacks.append(noam)

    n_dataset = int(subprocess.run(['wc', '-l', dataset_file],
                                   stdout=subprocess.PIPE).stdout.split()[
                        0].strip()) - 1

    # Training procedure #
    model.fit_generator(da.generator(),
                        floor(n_dataset / batch_size),
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=verbose)
