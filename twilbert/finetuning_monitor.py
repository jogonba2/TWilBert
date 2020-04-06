from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report)
from scipy.stats import pearsonr
from .finetuning_metrics import *
import numpy as np


class FinetuningMonitor:

    def __init__(self, monitor_metric="f1", average="macro",
                 class_metric=None, stance=False,
                 multi_label=False):
        self.step = 0
        self.best_step = -1
        self.best_value = -1
        self.monitor_metric = monitor_metric
        self.average = average
        self.class_metric = class_metric
        self.stance = stance
        self.multi_label = multi_label

    def __step__(self, truths, preds):
        best = False

        if self.multi_label:
            res = {"accuracy": jaccard_acc(np.array(truths),
                                           np.array(preds))}
        else:
            res = {"accuracy": accuracy_score(truths, preds),
                   "precision": precision_score(truths,
                                                preds,
                                                average=self.average),
                   "recall": recall_score(truths, preds,
                                          average=self.average),
                   "f1": f1_score(truths, preds,
                                  average=self.average),
                   "pearson": pearsonr(truths, preds)[0]}

        if self.stance:
            res["f1"] = mf1_stance(truths, preds)

        if self.class_metric:
            val = res[self.monitor_metric][self.class_metric]
        else:
            val = res[self.monitor_metric]

        if val > self.best_value:
            self.best_value = val
            self.best_step = self.step
            self.report_classification(res, "dev")
            best = True

        self.step += 1
        return best, res

    def __test_step__(self, truths, preds):

        if self.multi_label:
            res = {"accuracy": jaccard_acc(np.array(truths),
                                           np.array(preds))}
        else:
            res = {"accuracy": accuracy_score(truths, preds),
                   "precision": precision_score(truths,
                                                preds,
                                                average=None),
                   "recall": recall_score(truths, preds,
                                          average=None),
                   "f1": f1_score(truths, preds, average=None),
                   "macro-precision": precision_score(truths,
                                                      preds,
                                                      average="macro"),
                   "macro-recall": recall_score(truths,
                                                preds,
                                                average="macro"),
                   "macro-f1": f1_score(truths, preds,
                                        average="macro"),
                   "pearson": pearsonr(truths, preds)[0]}
        if self.stance:
            res["macro-f1"] = mf1_stance(truths, preds)

        self.report_classification(res, "test")

        return res

    def report_classification(self, res, sample_set):
        if sample_set == "dev":
            print("\n\n", "Best at dev, epoch %d\n"
                  % self.step + "-" * 20 + "\n")
        else:
            print("\n\n", "Best model evaluated on test\n" + "-" * 20 + "\n")

        if self.multi_label:
            print("1) Accuracy: %f" % res["accuracy"])
            print("\n" + "-" * 20 + "\n")

        else:
            print("1) Accuracy: %f" % res["accuracy"])
            print("2) F1: %s" % (str(res["f1"])))
            print("3) Precision: %s"
                  % (str(res["precision"])))
            print("4) Recall: %s"
                  % (str(res["recall"])))
            if sample_set == "test":
                print("5) F1 (macro): %s" % (str(res["macro-f1"])))
                print("6) Precision (macro): %s"
                      % (str(res["macro-precision"])))
                print("7) Recall (macro): %s"
                      % (str(res["macro-recall"])))
            print("8) Pearson: %f" % res["pearson"])
            print("\n" + "-" * 20 + "\n")
