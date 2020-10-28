import warnings
import os
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
###################################################################
import numpy as np
import tensorflow as tf
import random as rn
#np.random.seed(42)
#rn.seed(12345)
#from keras import backend as K
#tf.set_random_seed(1234)
####################################################################
from twilbert.models.bert import BertModel
from twilbert.preprocessing.tokenizer import FullTokenizer
from twilbert.optimization.lr_annealing import Noam
from twilbert.utils.utils import Utils as ut
from twilbert.utils.generator import SingleFinetuningGenerator
from twilbert.models.finetuning_models import finetune_ffn
from twilbert.utils.finetuning_monitor import FinetuningMonitor
from tqdm import trange
from itertools import product
from keras import backend as K
import json
import sys


if __name__ == "__main__":

    with open(sys.argv[1], "r") as json_file:
        config = json.load(json_file)

    # Dataset #

    train_file = config["dataset"]["train_file"]
    dev_file = config["dataset"]["dev_file"]
    test_file = config["dataset"]["test_file"]
    id_header = config["dataset"]["id_header"]
    text_header = config["dataset"]["text_header"]
    class_header = config["dataset"]["class_header"]
    delimiter = config["dataset"]["delimiter"]
    ###########

    # Task CSV Headers, Categories, Unbalancing and Number of Classes #

    categories = config["task"]["categories"]

    if "class_weights" in config["task"]:

        if type(config["task"]["class_weights"]) == dict:
            class_weight = {int(k): v
                            for (k, v) in config["task"]["class_weights"].items()}
        else:
            class_weight = config["task"]["class_weights"]

    n_classes = len(categories)
    metric = config["task"]["eval_metric"]
    average_metric = config["task"]["average_metric"]
    class_metric = config["task"]["class_metric"]
    stance_f1 = config["task"]["stance_f1"]
    multi_label = config["task"]["multi_label"]

    #######################

    # Representation #

    vocab_file = config["dataset"]["vocab_file"]
    tokenizer = FullTokenizer(vocab_file)
    vocab_size = len(tokenizer.vocab)
    max_len = config["representation"]["max_len"]
    bucket_min = config["representation"]["bucket_min"]
    bucket_max = config["representation"]["bucket_max"]
    bucket_steps = config["representation"]["bucket_steps"]
    preprocessing = config["representation"]["preprocessing"]

    ##################


    # Grid Search parameters #
    batch_size = config["grid_search"]["batch_size"]
    collapse_mode = config["grid_search"]["collapse_mode"]
    lr = config["grid_search"]["lr"]
    runs = config["grid_search"]["runs"]
    mon_metric = config["grid_search"]["monitorize"]

    # Finetuning model parameters #

    pred_batch_size = config["finetuning"]["pred_batch_size"]
    epochs = config["finetuning"]["epochs"]
    trainable_layers = config["finetuning"]["trainable_layers"]
    finetune_dropout = config["finetune_model"]["dropout"]
    loss = config["finetuning"]["loss"]
    save_model_path = config["finetuning"]["path_save_weights"]
    task_model_name = config["finetuning"]["model_name"]
    pretrained_model_weights = config["finetuning"]["path_load_weights"]
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
    pkm_params = config["model"]["pkm_params"]

    use_rop = config["model"]["rop"]["use_rop"]
    rop_n_hidden = config["model"]["rop"]["n_hidden"]
    rop_hidden_size = config["model"]["rop"]["hidden_size"]

    output_encoder_size = [hidden_size for i in range(n_encoders)]
    attention_size = [attention_size for i in range(n_encoders)]
    n_heads = [n_heads for i in range(n_encoders)]

    ##################################

    # Load Data #
    ids_tr, x_tr, y_tr = ut.load_dataset(train_file, id_header,
                                         text_header, class_header,
                                         categories, multi_label,
                                         delimiter)
    ids_dv, x_dv, y_dv = ut.load_dataset(dev_file, id_header,
                                         text_header, class_header,
                                         categories, multi_label,
                                         delimiter)
    ids_ts, x_ts, y_ts = ut.load_dataset(test_file, id_header,
                                         text_header, class_header,
                                         categories, multi_label,
                                         delimiter)

    if multi_label:
        n_classes = len(y_tr[0])

    gen_tr = SingleFinetuningGenerator(tokenizer, n_classes, bucket_min,
                                       bucket_max, bucket_steps,
                                       preprocessing, multi_label)
    gen_dv = SingleFinetuningGenerator(tokenizer, n_classes, bucket_min,
                                       bucket_max, bucket_steps,
                                       preprocessing, multi_label)
    gen_ts = SingleFinetuningGenerator(tokenizer, n_classes, bucket_min,
                                       bucket_max, bucket_steps,
                                       preprocessing, multi_label)

    n_buckets = int((bucket_max - bucket_min) / bucket_steps)

    tr_gen = gen_tr.generator(ids_tr, x_tr, y_tr)
    dv_gen = gen_dv.generator(ids_dv, x_dv, y_dv)
    ts_gen = gen_ts.generator(ids_ts, x_ts, y_ts)

    if not multi_label:
        if class_weight == "auto":
            class_weight = ut.max_ratio_weights(y_tr)
        elif class_weight == "ones":
            class_weight = None

    grid_params = list(product(batch_size, collapse_mode, lr))

    h_test_res, h_dev_res = {}, {}
    for (batch_size, collapse_mode, lr) in grid_params:

        print("Batch size: %d\nCollapse mode: %s\nLR: %f" %
             (batch_size, collapse_mode, lr))
        h_test_res[(batch_size, collapse_mode, lr)] = {}
        h_dev_res[(batch_size, collapse_mode, lr)] = {}
        for it in range(runs):
            K.clear_session()
            print("Run: %d" % it)
            # Load TWilBert model #
            twilbert_model = BertModel(max_len, vocab_size, embedding_size,
                                       output_encoder_size, attention_size,
                                       n_heads, cross_sharing,
                                       factorize_embeddings,
                                       input_dropout, output_dropout,
                                       rop_n_hidden, rop_hidden_size,
                                       optimizer, accum_iters, pkm, pkm_params,
                                       input_length=None, use_rop=use_rop)

            twilbert_model.build()

            model = twilbert_model.model
            pretrained_model = twilbert_model.pretrained_model
            twilbert_model.compile(model)
            model.load_weights(pretrained_model_weights)
            #########################

            # Load finetune model #
            finetune_model = finetune_ffn(pretrained_model, n_classes,
                                          trainable_layers, collapse_mode,
                                          finetune_dropout=finetune_dropout,
                                          loss=loss, lr=lr, multi_label=multi_label,
                                          optimizer=optimizer, accum_iters=accum_iters)

            monitor = FinetuningMonitor(metric, average_metric,
                                        class_metric, stance_f1, multi_label)
            callbacks = None

            if noam_annealing:
                noam = Noam(warmup_steps=warmup_steps,
                            hidden_dims=hidden_size,
                            accum_iters=accum_iters)
                callbacks = [noam]

            for e in range(epochs):
                avg_loss = 0.
                avg_acc = 0.
                seen_buckets = n_buckets
                for b in range(n_buckets):
                    (bx, by) = next(tr_gen)
                    if len(bx[0]) == 0:
                        seen_buckets -= 1
                        continue
                    hist = finetune_model.fit(x=bx, y=by,
                                              epochs=1, verbose=0,
                                              batch_size=batch_size,
                                              class_weight=class_weight,
                                              callbacks=callbacks)
                    avg_loss += (hist.history["loss"][0] / seen_buckets)
                    avg_acc += (hist.history["acc"][0] / seen_buckets)


                dev_preds = []
                dev_truths = []
                for b in range(n_buckets):
                    (bx, by) = next(dv_gen)
                    if len(bx[0]) == 0:
                        continue

                    if not multi_label:
                        dev_truths += by.argmax(axis=-1).tolist()
                    else:
                        dev_truths += by.tolist()

                    preds = finetune_model.predict(x=bx, batch_size=pred_batch_size)
                    if not multi_label:
                        preds = preds.argmax(axis=-1)
                    preds = preds.tolist()
                    dev_preds += preds
                best, act_dev_res = monitor.__step__(dev_truths, dev_preds)

                if best:
                    test_preds = []
                    test_truths = []
                    for b in range(n_buckets):
                        (bx, by) = next(ts_gen)
                        if len(bx[0]) == 0:
                            continue

                        if not multi_label:
                            test_truths += by.argmax(axis=-1).tolist()
                        else:
                            test_truths += by.tolist()

                        preds = finetune_model.predict(x=bx,
                                                       batch_size=pred_batch_size)
                        if not multi_label:
                            preds = preds.argmax(axis=-1)
                        preds = preds.tolist()
                        test_preds += preds
                    best_test_res = monitor.__test_step__(test_truths, test_preds)
                    best_dev_res = act_dev_res
                    model_json = finetune_model.to_json()
                    with open(save_model_path + "/" +
                              task_model_name + ".json", "w") as json_file:
                        json_file.write(model_json)
                    finetune_model.save(
                        save_model_path + "/" + task_model_name + ".h5")

            # Añado a los diccionarios de dev y test los parámetros y la iteración donde se ha maximizado
            # la métrica de evaluación
            h_dev_res[(batch_size, collapse_mode, lr)][it] = best_dev_res
            h_test_res[(batch_size, collapse_mode, lr)][it] = best_test_res

    # Coger el mejor en validación de todos los parámetros y todas las iteraciones #
    best_params, best_it = ut.get_best_params(h_dev_res, mon_metric)

    print("Best params:", best_params)
    # Resultados en dev y test del modelo que maximiza la métrica en validación
    print("Dev results:", h_dev_res[best_params][best_it])
    print("Test results:", h_test_res[best_params][best_it])
