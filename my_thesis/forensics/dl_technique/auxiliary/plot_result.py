import numpy as np
import matplotlib.pyplot as plt
import math

import os.path as osp
import csv
from typing import List

from torch import AnyType

NUM_INFO = 17
INTERVAL = 200
K_SPLINE_LOSS_ACC = 2
K_SPLINE_METRIC = 2

def read_file(csv_file: str):
    """ 
    0       1           2               3           4               5               6               7                   8               9               10              11              12                  13              14                  15              16          
    epoch, Train loss, Train accuracy, Test loss, Test accuracy, Test real pre, Test real rec, Test real F1-Score, Test fake pre, Test fake rec, Test fake F1-Score, Test micro pre, Test micro rec, Val micro F1-Score, Test macro pre, Test macro rec, Test macro F1-Score
    epoch, Train loss, Train accuracy, Val loss,  Val accuracy,  Val real pre,  Val real rec,  Val real F1-Score,  Val fake pre,  Val fake rec,  Val fake F1-Score,  Val micro pre,  Val micro rec,  Val micro F1-Score, Val macro pre,  Val macro rec,  Val macro F1-Score
    Args:
        csv_file (str): path to file
    """
    global NUM_INFO
    result = [[] for _ in range(NUM_INFO)]
    # Read file
    file = open(csv_file)
    csvreader = csv.reader(file)
    # Get result:
    for idx, row in enumerate(csvreader):
        if idx == 0:
            continue
        for i in range(NUM_INFO):
            if i == 0:
                result[0].append(int(row[0]))
                continue
            result[i].append(float(row[i]))
    
    if 'test' in osp.basename(csv_file):
        phase = "Test"
    else:
        phase = "Validation"
        
    if 'epoch' in csv_file:
        iter_method = 'Epoch'
    else:
        iter_method = 'Step'
    
    results = []
    for i, r in enumerate(result):
        results.append(np.array(r, dtype=np.int32 if i == 0 else np.float32))
    return results, phase, iter_method

def plot_loss_and_accuracy(iter_loop: List[int], iter_method: str, values: List[List[float]], phase="Val"):
    """ values: [train_loss, train_acc, val_loss, val_acc]
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(iter_loop, values[0], label="Train loss")
    plt.plot(iter_loop, values[2], label=phase + " loss")
    plt.legend()
    plt.title("Loss per " + iter_method)
    # plt.xticks([i for i in range(1, len(epochs)+1)])
    # plt.yticks([0.1*i for i in range(1, 16)])
    
    plt.subplot(1, 2, 2)
    plt.plot(iter_loop, values[1], label="Train accuracy")
    plt.plot(iter_loop, values[3], label=phase + " accuracy")
    plt.legend()
    plt.title("Accuracy per " + iter_method)
    plt.plot()
    plt.savefig("result/Loss-Accuracy {} per {}.png".format(phase, iter_method))
    
def plot_metrics(iter_loop: List[int], iter_method: str, values: List[List[List[float]]], phase="Val"):
    """ values: [[pre, rec, f1] * [real, fake, micro, macro]]
    """
    plt.figure(figsize=(20, 10))
    assert len(values) == 4, "False parameter!"
    
    # i: 0 -> 3: real, fake, micro, macro
    title = ["Class real's metrics", "Class fake's metrics", "Micro average metrics", "Macro average metrics"]
    for i in range(0, 4):
        plt.subplot(2, 2, i+1)
        # j: 0 -> 2: precision, recall, f1-score
        plt.plot(iter_loop, values[i][0], label="Precision")
        plt.plot(iter_loop, values[i][1], label="Recall")
        plt.plot(iter_loop, values[i][2], label="F1-Score")
        plt.legend()
        plt.title(title[i])
    plt.plot()
    plt.savefig("result/Metrics {} per {}.png".format(phase, iter_method))

def smooth_curve_with_pyplot(results: List[List[AnyType]], iter_method: str, phase='Val'):
    """results:
        0       1           2               3           4               5               6               7                   8               9               10              11              12                  13              14                  15              16          
        epoch, Train loss, Train accuracy, Test loss, Test accuracy, Test real pre, Test real rec, Test real F1-Score, Test fake pre, Test fake rec, Test fake F1-Score, Test micro pre, Test micro rec, Val micro F1-Score, Test macro pre, Test macro rec, Test macro F1-Score
    """
    from scipy.interpolate import make_interp_spline
    global NUM_INFO, INTERVAL, K_SPLINE_LOSS_ACC, K_SPLINE_METRIC
    
    iter_loop = results[0]
    iter_loop_new = np.linspace(iter_loop.min(), iter_loop.max(), INTERVAL)

    #define spline
    K = [-1] + [K_SPLINE_LOSS_ACC] * 4 + [K_SPLINE_METRIC] * 12
    spl = [make_interp_spline(iter_loop, results[i], k=K[i]) for i in range(1, NUM_INFO)]
    results_smooth = [-1] + [spl_item(iter_loop_new) for spl_item in spl]
    
    plot_loss_and_accuracy(iter_loop_new, iter_method, values=results_smooth[1:5], phase=phase)
    plot_metrics(iter_loop_new, iter_method, values=[results_smooth[5: 8], results_smooth[8: 11], results_smooth[11: 14], results_smooth[14: 17]], phase=phase)

def plot_graph(csv_file: str):
    if csv_file == '':
        return
    results, phase, iter_method = read_file(csv_file)
    smooth_curve_with_pyplot(results, iter_method, phase)
    
if __name__ == '__main__':
    dict = {}
    methods = ['meso4', 'xception', 'dual_efficient', 'dual_attn_efficient', 'dual_efficient_vit']
    datasets = ['uadfv', 'ff_all', 'celeb_df', 'dfdc', 'df_in_the_wild']
    plot_graph(dict[methods, datasets])