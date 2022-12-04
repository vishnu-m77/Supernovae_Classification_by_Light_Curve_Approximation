import numpy as np
from  sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_recall_curve, auc, recall_score, precision_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
from sklearn.utils import resample
import warnings


def gen_report(y_test, y_test_pred, n_iters=1000, decimals=3):
    
    metrics = []
    inds = np.arange(len(y_test))
    for i in range(n_iters):
        inds_boot = resample(inds)
        try:
            roc_auc = roc_auc_score(y_test[inds_boot], y_test_pred[inds_boot])
        except ValueError:
            roc_auc = 0
        try:
            logloss = log_loss(y_test[inds_boot], y_test_pred[inds_boot], eps=10**-6)
        except ValueError:
            logloss = 0
        accuracy = accuracy_score(y_test[inds_boot], 1 * (y_test_pred[inds_boot] > 0.5))
        precision, recall, _ = precision_recall_curve(y_test[inds_boot], y_test_pred[inds_boot])
        pr_auc = auc(recall, precision)
        recall = recall_score(y_test[inds_boot], 1 * (y_test_pred[inds_boot] > 0.5))
        precision = precision_score(y_test[inds_boot], 1 * (y_test_pred[inds_boot] > 0.5))
        metrics.append([roc_auc, pr_auc, logloss, accuracy, recall, precision])
    metrics = np.array(metrics)
    report = pd.DataFrame(columns=["ROC_AUC", 'PR-AUC', 'LogLoss', 'Accuracy', 'Recall', 'Precision'],
                          data=[metrics.mean(axis=0).round(decimals=4), metrics.std(axis=0).round(decimals=4)], 
                          index=['mean', 'std'])
    
    return report
