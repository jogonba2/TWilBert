from keras import backend as K
from sys import modules


def loss_indexer(loss_str):
    this = modules[__name__]
    if hasattr(this, loss_str):
        return getattr(this, loss_str)()
    else:
        return loss_str


def jaccard_acc():
    def jaccard(y_true, y_pred):
        bs = 16
        num = y_true * y_pred
        den = K.maximum(y_true, y_pred)
        sum_num = K.sum(num, axis=1)
        sum_den = K.sum(den, axis=1)
        quotient = (sum_num + 1e-16) / (sum_den + 1e-16)
        total_sum = K.sum(quotient)
        return -(total_sum / bs)
    return jaccard


def root_mean_squared_error():
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
    return rmse


def binary_f1(beta=1):
    def f_loss(y_true, y_pred):
        num = (1. + beta**2) * K.sum(y_pred * y_true)
        den = K.sum(y_pred + (beta**2 * y_true))
        return -(num / den)
    return f_loss


def macro_f1(epsilon=1e-16, beta=1):

    def f_loss(y_true, y_pred):
        tp = K.sum(y_true * y_pred, axis=0)
        fp = K.sum((1. - y_true) * y_pred, axis=0)
        fn = K.sum(y_true * (1. - y_pred), axis=0)
        pr = (tp / (tp + fp + epsilon) + epsilon)
        rc = (tp / (tp + fn + epsilon) + epsilon)
        f1 = ((1. + (beta**2)) * pr * rc) / ((beta**2) * pr + rc)
        return -K.mean(f1)

    return f_loss


def f1_per_class(epsilon=1e-16, beta=1):

    def f_loss(y_true, y_pred):
        tp = K.sum(y_true * y_pred, axis=0)
        fp = K.sum((1. - y_true) * y_pred, axis=0)
        fn = K.sum(y_true * (1. - y_pred), axis=0)
        pr = (tp / (tp + fp + epsilon) + epsilon)
        rc = (tp / (tp + fn + epsilon) + epsilon)
        f1 = ((1. + (beta**2)) * pr * rc) / ((beta**2) * pr + rc)
        loss = -K.sum(y_true * f1, axis=1)
        return loss

    return f_loss


def f1_macro(epsilon=1e-16, beta=1):

    def f_loss(y_true, y_pred):
        tp = K.sum(y_true * y_pred, axis=0)
        fp = K.sum((1. - y_true) * y_pred, axis=0)
        fn = K.sum(y_true * (1. - y_pred), axis=0)
        pr = (tp / (tp + fp + epsilon) + epsilon)
        rc = (tp / (tp + fn + epsilon) + epsilon)
        pr = K.mean(pr)
        rc = K.mean(rc)
        f1 = ((1. + (beta**2)) * pr * rc) / ((beta**2) * pr + rc)
        return -f1

    return f_loss


def macro_precision(epsilon=1e-16):

    def f_loss(y_true, y_pred):
        tp = K.sum(y_true * y_pred, axis=0)
        fp = K.sum((1. - y_true) * y_pred, axis=0)
        pr = (tp / (tp + fp + epsilon) + epsilon)
        return -K.mean(pr)

    return f_loss


def precision_per_class(epsilon=1e-16):

    def f_loss(y_true, y_pred):
        tp = K.sum(y_true * y_pred, axis=0)
        fp = K.sum((1. - y_true) * y_pred, axis=0)
        pr = (tp / (tp + fp + epsilon) + epsilon)
        return -K.sum(y_true * pr, axis=1)

    return f_loss


def macro_recall(epsilon=1e-16):

    def f_loss(y_true, y_pred):
        tp = K.sum(y_true * y_pred, axis=0)
        fn = K.sum(y_true * (1. - y_pred), axis=0)
        rc = (tp / (tp + fn + epsilon) + epsilon)
        return -K.mean(rc)

    return f_loss


def recall_per_class(epsilon=1e-16):

    def f_loss(y_true, y_pred):
        tp = K.sum(y_true * y_pred, axis=0)
        fn = K.sum(y_true * (1. - y_pred), axis=0)
        rc = (tp / (tp + fn + epsilon) + epsilon)
        return -K.sum(y_true * rc, axis=1)

    return f_loss


def micro_precision(epsilon=1e-16):

    def f_loss(y_true, y_pred):
        tp = K.sum(y_true * y_pred, axis=0)
        fp = K.sum((1. - y_true) * y_pred, axis=0)
        tps = K.sum(tp, axis=0)
        fps = K.sum(fp, axis=0)
        return -(tps / (tps + fps + epsilon) + epsilon)

    return f_loss


def micro_recall(epsilon=1e-16):

    def f_loss(y_true, y_pred):
        tp = K.sum(y_true * y_pred, axis=0)
        fn = K.sum(y_true * (1. - y_pred), axis=0)
        tps = K.sum(tp, axis=0)
        fns = K.sum(fn, axis=0)
        return -(tps / (tps + fns + epsilon) + epsilon)

    return f_loss


def micro_f1(epsilon=1e-16, beta=1):

    def f_loss(y_true, y_pred):
        tp = K.sum(y_true * y_pred, axis=0)
        fn = K.sum(y_true * (1. - y_pred), axis=0)
        fp = K.sum((1. - y_true) * y_pred, axis=0)
        tps = K.sum(tp, axis=0)
        fns = K.sum(fn, axis=0)
        fps = K.sum(fp, axis=0)
        pr = (tps / (tps + fps + epsilon) + epsilon)
        rc = (tps / (tps + fns + epsilon) + epsilon)
        f1 = ((1. + (beta**2)) * pr * rc) / ((beta**2) * pr + rc)
        return -f1

    return f_loss
