import numpy as np
import torch
from typing import List
from utils.util import save_list_to_file

def accuracy_score(y_label: np.ndarray, y_pred_label: np.ndarray):
    assert len(y_pred_label) == len(y_label), "Prediction and groundtruth not have the same length."
    corrected = np.sum(y_label == y_pred_label).astype(np.float64)
    return 1.0 * corrected / len(y_label)

def precision_score(y_label: np.ndarray, y_pred_label: np.ndarray, average='binary', pos_label=1):
    score = -1
    if average == 'binary':
        tp = np.sum((y_label == pos_label) * (y_pred_label == pos_label))
        fp = np.sum((y_label == 1 - pos_label) * (y_pred_label == pos_label))
        score = 1.0 * tp / (tp + fp)
    elif average == 'micro':
        tp = np.sum(y_label == y_pred_label)
        fp = np.sum(y_label != y_pred_label)
        score = 1.0 * tp / (tp + fp)

    else:   # 'macro'
        score_0 = precision_score(y_label, y_pred_label, pos_label=0)
        score_1 = precision_score(y_label, y_pred_label, pos_label=1)
        score = (score_0 + score_1) / 2
    return score

def recall_score(y_label: np.ndarray, y_pred_label: np.ndarray, average='binary', pos_label=0):
    score = -1
    if average == 'binary':
        tp = np.sum((y_label == pos_label) * (y_pred_label == pos_label))
        fn = np.sum((y_label == pos_label) * (y_pred_label == 1 - pos_label))
        score = 1.0 * tp / (tp + fn)
    elif average == 'micro':
        tp = np.sum(y_label == y_pred_label)
        fn = np.sum(y_label != y_pred_label)
        score = 1.0 * tp / (tp + fn)

    else:   # 'macro'
        score_0 = recall_score(y_label, y_pred_label, pos_label=0)
        score_1 = recall_score(y_label, y_pred_label, pos_label=1)
        score = (score_0 + score_1) / 2
    return score

def f1_score(y_label: np.ndarray, y_pred_label: np.ndarray, average='binary', pos_label=0):
    score = -1
    if average == 'binary':
        precision = precision_score(y_label, y_pred_label, average='binary', pos_label=pos_label)
        rec = recall_score(y_label, y_pred_label, average='binary', pos_label=pos_label)
        score = 2.0 * precision * rec / (precision + rec)
    elif average == 'micro':
        precision = precision_score(y_label, y_pred_label, average='micro')
        rec = recall_score(y_label, y_pred_label, average='micro')
        score = 2.0 * precision * rec / (precision + rec)
    else:
        f1_0 = f1_score(y_label, y_pred_label, average='binary', pos_label=0)
        f1_1 = f1_score(y_label, y_pred_label, average='binary', pos_label=1)
        score = (f1_0 + f1_1)/2
    return score

def calculate_cls_metric(y_label: np.ndarray, y_pred_label: np.ndarray, average='binary', pos_label=1):
    precision = precision_score(y_label, y_pred_label, average=average, pos_label=pos_label)
    recall = recall_score(y_label, y_pred_label, average=average, pos_label=pos_label)
    f1 = f1_score(y_label, y_pred_label, average=average, pos_label=pos_label)
    return precision, recall, f1

def calculate_cls_metrics(y_label: np.ndarray, y_pred_label: np.ndarray, save=False, print_metric=False):   
    accuracy = accuracy_score(y_label, y_pred_label)
    mic_pre, mic_rec, mic_f1 = calculate_cls_metric(y_label, y_pred_label, average='micro')
    mac_pre, mac_rec, mac_f1 = calculate_cls_metric(y_label, y_pred_label, average='macro')
    real_pre, real_rec, real_f1 = calculate_cls_metric(y_label, y_pred_label, average='binary', pos_label=0)
    fake_pre, fake_rec, fake_f1 = calculate_cls_metric(y_label, y_pred_label, average='binary', pos_label=1)

    if 0 in [accuracy, mic_pre, mic_rec, mic_f1, mac_pre, mac_rec, mac_f1, real_pre, real_rec, real_f1, fake_pre, fake_rec, fake_f1] and save:
        print("Bug appears!")
        save_list_to_file(y_label, file_name='y_label.txt', overwrite=True)
        save_list_to_file(y_pred_label, file_name='y_pred_label.txt', overwrite=True)
        
    if print_metric:
        result = 'Customize metrics: '
        result += "acc: {:.4f} --- ".format(accuracy)
        result += "real f1-score: {:.4f} --- ".format(real_f1)
        result += "fake f1-score: {:.4f} --- ".format(fake_f1)
        result += "avg f1-score: {:.4f} --- ".format(mac_f1)
        print(result)
    return accuracy, (real_pre, real_rec, real_f1),\
                     (fake_pre, fake_rec, fake_f1),\
                     (mic_pre, mic_rec, mic_f1),\
                     (mac_pre, mac_rec, mac_f1)
                     
if __name__ == '__main__':
    f1_score(np.array([1.0]), np.array([0.]))