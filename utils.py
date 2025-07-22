import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
import logging
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

logging.basicConfig(
    level=logging.WARN,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Metric(y_total, pred_total):
    y_total_array = np.array(y_total.to('cpu'))
    pred_total_array = np.array(pred_total.to('cpu'))

    cm = confusion_matrix(y_total_array, pred_total_array)

    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]
    FAR = round(FP / (TN + FP), 4)
    DR = round(TP / (TP + FN), 4)


def Metric_process(y_total, pred_total):
    y_total_array = np.array(y_total.to('cpu'))
    pred_total_array = np.array(pred_total.to('cpu'))

    cm = confusion_matrix(y_total_array, pred_total_array)

    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    Precision = round(TP / (TP + FP), 4)
    FAR = round(FP / (TN + FP), 4)
    DR = round(TP / (TP + FN), 4)
    F1 = round(2 * Precision * DR / (Precision + DR), 4)

    return DR, FAR, F1


def ROC(test_y_total, test_y_pred_total_roc, alg, dataset, file):
    test_y_total = np.array(test_y_total.to('cpu'))
    test_y_pred_total_roc = np.array(test_y_pred_total_roc.to('cpu'))

    auc = roc_auc_score(test_y_total, test_y_pred_total_roc[:, 1])
    fpr, tpr, thresholds = roc_curve(test_y_total, test_y_pred_total_roc[:, 1])

    roc_df = pd.concat([pd.DataFrame(fpr), pd.DataFrame(tpr), pd.DataFrame(np.array([auc, alg]))], axis=1)

    roc_df.to_csv("./roc/" + alg + "_" + dataset + "_" + file + ".csv", header=False, index=False)

    plt.plot(fpr, tpr, label='ROC')

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()