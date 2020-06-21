# This file is inspired / taken from Abhishek Thakur's tutorial on Applied Machine Learning
# Link : https://github.com/abhishekkrthakur/mlframework/blob/master/src/metrics.py
# Please consider starring the repository to appreciate the developer

from sklearn import metrics as skmetrics
import numpy as np

"""
This is a helper class to get a shortcut to all the evaluation metrics 
Includes:
> Accuracy
> F1 score
> Precision score
> Recal score
> ROC-AUC score
> Logloss

For the purpose of this competition we will use the recall_score
"""

class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            "accuracy" : self._accuracy,
            "f1" : self._f1,
            "recall" : self._recall,
            "precision" : self._precision,
            "auc" : self._auc,
            "logloss" : self._logloss
        }
    

    def __call__(self, metric, y_true, y_pred, y_proba=None):
        if metric not in self.metrics:
            raise Exception(f"Metric {metric} not implemented")
    
        if metric=="auc":
            if y_proba is not None:
                return self._auc(y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be none for auc")
        
        if metric=="logloss":
            if y_proba is not None:
                return self._logloss(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be none for logloss")
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)
    

    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    

    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)
    

    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)
    

    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)
    

    @staticmethod
    def _auc(y_true, y_pred):
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)
    

    @staticmethod
    def _logloss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)