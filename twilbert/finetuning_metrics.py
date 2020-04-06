from sklearn.metrics import classification_report
import numpy as np


def jaccard_acc(y_true, y_pred):
    num = y_true * y_pred
    den = np.maximum(y_true, y_pred)
    sum_num = np.sum(num, axis=1)
    sum_den = np.sum(den, axis=1)
    quotient = (sum_num + 1e-16) / (sum_den + 1e-16)
    total_sum = np.sum(quotient)
    return total_sum / len(y_true)

def mf1_stance(truths, preds):
    s = classification_report(truths, preds, digits=5)
    aux = s.split("\n")
    f1_against = float(aux[2].split("   ")[6])
    f1_favor = float(aux[4].split("   ")[6])
    return (f1_against + f1_favor) / 2.
