from keras.layers import (Input, Dense, Lambda,
                          GlobalAveragePooling1D, GlobalMaxPooling1D,
                          Dropout, Concatenate)
from keras.models import Model
from twilbert.finetuning_losses import loss_indexer
from twilbert.optimization.optimizers import *
from keras.initializers import TruncatedNormal as tn


def finetune_ffn(pretrained_model, n_classes,
                 trainable_layers="all", collapse_mode="cls",
                 finetune_dropout=0.15,
                 loss="categorical_crossentropy",
                 init_range=0.02, lr=0.001,
                 multi_label=False,
                 optimizer="adam", accum_iters=1):

    assert collapse_mode in ["cls", "max", "avg", "concat"]
    if trainable_layers != "all":
        assert type(trainable_layers) == list
        model_layers = []
        for layer in pretrained_model.layers:
            layer.trainable = False
            if "embedding" in layer.name or "encoder" in layer.name:
                model_layers.append(layer)
        for k in trainable_layers:
            model_layers[k].trainable = True

    input_tokens = Input(shape=(None,))
    input_positions = Input(shape=(None,))
    input_segments = Input(shape=(None,))

    pretrained_output = pretrained_model([input_tokens,
                                          input_positions,
                                          input_segments])

    if collapse_mode == "cls":
        cls_output = Lambda(lambda x: x[:, 0, :])(pretrained_output)

    else:

        if collapse_mode == "avg":
            cls_output = GlobalAveragePooling1D()(pretrained_output)
        elif collapse_mode == "max":
            cls_output = GlobalMaxPooling1D()(pretrained_output)
        elif collapse_mode == "concat":
            avg = GlobalAveragePooling1D()(pretrained_output)
            mx = GlobalMaxPooling1D()(pretrained_output)
            cls = Lambda(lambda x: x[:, 0, :])(pretrained_output)
            cls_output = Concatenate(axis=-1)([cls, avg, mx])

    cls_output = Dropout(finetune_dropout)(cls_output)

    if not multi_label:
        output = Dense(n_classes, activation="softmax",
                       kernel_initializer=tn(init_range))(cls_output)

    else:
        output = Dense(n_classes, activation="sigmoid",
                       kernel_initializer=tn(init_range))(cls_output)

    finetune_model = Model(inputs=[input_tokens,
                                   input_positions,
                                   input_segments],
                           outputs=output)

    if optimizer == "adam_accumulated":
        opt = ADAM(lr=lr, accum_iters=accum_iters)
    elif optimizer == "lamb_accumulated":
        opt = LAMB(lr=lr, accum_iters=accum_iters)
    else:
        opt = optimizer

    loss = loss_indexer(loss)
    finetune_model.compile(optimizer=opt,
                           loss=loss,
                           metrics=["accuracy"])

    return finetune_model
