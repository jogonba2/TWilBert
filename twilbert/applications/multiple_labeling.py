import warnings
import os
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from twilbert.models.twilbert_model import TWilBertModel
from twilbert.preprocessing.tokenizer import FullTokenizer
from twilbert.optimization.lr_annealing import Noam
from twilbert.utils.utils import Utils as ut
from twilbert.utils.generator import MultipleFinetuningGenerator
from twilbert.models.finetuning_models import finetune_ffn
from twilbert.utils.finetuning_monitor import FinetuningMonitor
from tqdm import trange
import json
import sys

if __name__ == "__main__":

    with open(sys.argv[1], "r") as json_file:
        config = json.load(json_file)
    ut.set_random_seed()

    # Dataset #

    test_file = config["dataset"]["test_file"]
    output_file = config["dataset"]["output_file"]
    id_header = config["dataset"]["id_header"]
    text_header = config["dataset"]["text_header"]
    aux_header = config["dataset"]["aux_header"]
    delimiter = config["dataset"]["delimiter"]
    ###########

    # Task CSV Headers, Categories, Unbalancing and Number of Classes #

    rev_categories = config["task"]["rev_categories"]
    n_classes = len(rev_categories)
    #######################

    # Representation #

    vocab_file = config["dataset"]["vocab_file"]
    tokenizer = FullTokenizer(vocab_file)
    vocab_size = len(tokenizer.vocab)
    max_len = config["representation"]["max_len"]
    bucket_min_a = config["representation"]["bucket_min_a"]
    bucket_max_a = config["representation"]["bucket_max_a"]
    bucket_min_b = config["representation"]["bucket_min_b"]
    bucket_max_b = config["representation"]["bucket_max_b"]
    bucket_steps = config["representation"]["bucket_steps"]
    preprocessing = config["representation"]["preprocessing"]

    ##################

    # Finetuning model parameters #

    batch_size = config["finetuning"]["batch_size"]
    pred_batch_size = config["finetuning"]["pred_batch_size"]
    epochs = config["finetuning"]["epochs"]
    trainable_layers = config["finetuning"]["trainable_layers"]
    collapse_mode = config["finetuning"]["collapse_mode"]
    finetune_dropout = config["finetuning"]["dropout"]
    loss = config["finetuning"]["loss"]
    lr = config["finetuning"]["lr"]
    finetune_model_weights = config["finetuning"]["path_load_finetuned_weights"]
    optimizer = config["finetuning"]["optimizer"]
    accum_iters = config["finetuning"]["accum_iters"]
    noam_annealing = config["finetuning"]["noam_annealing"]
    warmup_steps = config["finetuning"]["warmup_steps"]

    ##############################

    # Pretrained TWilBert parameters #

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

    encoder_size = [hidden_size for i in range(n_encoders)]
    attention_size = [attention_size for i in range(n_encoders)]
    n_heads = [n_heads for i in range(n_encoders)]

    ##################################

    # Load TWilBert model #

    twilbert_model = TWilBertModel(max_len, vocab_size, embedding_size,
                                   encoder_size, attention_size,
                                   n_heads, cross_sharing,
                                   factorize_embeddings,
                                   input_dropout, output_dropout,
                                   rop_n_hidden, rop_hidden_size,
                                   optimizer, accum_iters, pkm, pkm_params,
                                   input_length=None)

    twilbert_model.build()

    model = twilbert_model.model
    pretrained_model = twilbert_model.pretrained_model
    twilbert_model.compile(model)

    #########################

    # Load Data #
    ids_ts, x1_ts, x2_ts = ut.load_test_multiple_dataset(test_file,
                                                         id_header,
                                                         text_header,
                                                         aux_header,
                                                         delimiter)

    gen_ts = MultipleFinetuningGenerator(tokenizer, 1, bucket_min_a,
                                         bucket_min_b, bucket_max_a,
                                         bucket_max_b, bucket_steps,
                                         preprocessing)

    n_buckets = int(((bucket_max_a - bucket_min_a) / bucket_steps) ** 2)

    ts_gen = gen_ts.generator(ids_ts, x1_ts, x2_ts,
                              [0 for i in range(len(ids_ts))])

    # Load finetune model #
    finetune_model = finetune_ffn(pretrained_model, n_classes,
                                  trainable_layers, collapse_mode,
                                  finetune_dropout=finetune_dropout,
                                  loss=loss, lr=lr,
                                  optimizer=optimizer, accum_iters=accum_iters)

    finetune_model.load_weights(finetune_model_weights)
    print(finetune_model.summary())

    test_preds = []
    test_ids = []
    for b in range(n_buckets):
        (bi, bx, by) = next(ts_gen)
        if len(bx[0]) == 0:
            continue
        preds = finetune_model.predict(x=bx, batch_size=pred_batch_size)
        preds = preds.argmax(axis=-1).tolist()
        test_preds += preds
        test_ids += bi

    # Save results #
    with open(output_file, "w") as fw:
        for i in range(len(test_preds)):
            fw.write(test_ids[i] + "\t" + rev_categories[str(test_preds[i])] + "\n")
