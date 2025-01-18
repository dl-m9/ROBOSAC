import torch
from torch.utils.data import DataLoader
import os
from train import CPGuardDataset, detector, init_weights
from sklearn.metrics import precision_score, recall_score, f1_score
from models import ResNetBinaryClassifier

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from test import test_model





if __name__ == '__main__':

    leave_one_outs = ['pgd', 'bim', 'cw-l2', 'fgsm','GN']

    for leave_one_out in leave_one_outs:
        weights_dirs = os.listdir('/data2/user2/senkang/CP-GuardBench/cpguard/logs')
        current_weights_dir = [dir for dir in weights_dirs if dir.endswith(leave_one_out)][0]
        pth_path = os.path.join('/data2/user2/senkang/CP-GuardBench/cpguard/logs', current_weights_dir, '10.pth')



        base_dir = '/data2/user2/senkang/CP-GuardBench/CP-GuardBench_RawData'
        full_path = os.path.join(base_dir, f'test_{leave_one_out}_0.2')
        # full_path = os.path.join(base_dir, 'test')


        results, labels, predictions = test_model(full_path, pth_path)
        

        # print(f'Results (leave {leave_one_out} out):')
        # for metric, value in results.items():
        #     print(f"{metric}: {value:.4f}")