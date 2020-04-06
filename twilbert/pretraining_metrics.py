from keras import backend as K
import tensorflow as tf


def acc_m(y_true, y_pred):
    y_pred_ = K.cast(K.argmax(y_pred, axis=-1), 'int32')
    y_true_ = K.cast(K.squeeze(y_true, axis=-1), "int32")
    mask = K.cast(K.any(y_true, axis=-1), "int32")
    correct = K.cast(K.equal(y_true_, y_pred_), 'int32')
    flattened = K.flatten(correct)
    mask = K.flatten(mask)
    flattened = flattened * mask
    length = K.sum(mask)
    acc = tf.reduce_sum(flattened) / length
    return acc


def acc_r(y_true, y_pred):
    if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.squeeze(y_true, -1)
    y_pred_labels = K.argmax(y_pred, axis=-1)
    y_pred_labels = K.cast(y_pred_labels, K.floatx())
    return K.cast(K.equal(y_true, y_pred_labels), K.floatx())
