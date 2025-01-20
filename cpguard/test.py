import torch
from torch.utils.data import DataLoader
import os
from train import CPGuardDataset, detector, init_weights
from sklearn.metrics import precision_score, recall_score, f1_score
from models import ResNetBinaryClassifier

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse


def plot_roc_curve(all_labels, all_predictions, save_path=None):
    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(4, 3), dpi=300)
    plt.grid(linestyle='--')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def test_model(eval_data_dir, pretrained_weights_path):
    if_residual = False
    eval_dataset = CPGuardDataset(eval_data_dir, if_residual=if_residual)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

    model = ResNetBinaryClassifier().to('cuda')
    model.load_state_dict(torch.load(pretrained_weights_path))
    model.to('cuda')
    model.eval()

    criterion = torch.nn.BCELoss()

    eval_loss = 0.0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []

    from tqdm import tqdm

    with torch.no_grad():
        for i, (data, file_name) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc=f"Evaluating {eval_data_dir.split('/')[-1]}"):
            residual_latent_feature_list = torch.stack([item[0] for item in data]).squeeze().to('cuda')
            label_list = torch.tensor([item[1] for item in data], dtype=torch.float).to('cuda')

            outputs, _ = model(residual_latent_feature_list)
            outputs = outputs.squeeze() 
            loss = criterion(outputs, label_list)
            eval_loss += loss.item()

            predicted = (outputs > 0.5).float()
            
            true_positives += ((predicted == 1) & (label_list == 1)).sum().item()
            true_negatives += ((predicted == 0) & (label_list == 0)).sum().item()
            false_positives += ((predicted == 1) & (label_list == 0)).sum().item()
            false_negatives += ((predicted == 0) & (label_list == 1)).sum().item()
            
            total_predictions += label_list.size(0)

            all_labels.extend(label_list.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())  # Use raw outputs for ROC curve

    avg_eval_loss = eval_loss / len(eval_dataloader)
    accuracy = (true_positives + true_negatives) / total_predictions
    true_positive_rate = true_positives / (true_positives + false_negatives)
    false_positive_rate = false_positives / (false_positives + true_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f'Results for {eval_data_dir.split("/")[-1]}:')
    print(f'Evaluation Loss: {avg_eval_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'True Positive Rate: {true_positive_rate:.4f}')
    print(f'False Positive Rate: {false_positive_rate:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print()

    return {
        'loss': avg_eval_loss,
        'accuracy': accuracy,
        'tpr': true_positive_rate,
        'fpr': false_positive_rate,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }, all_labels, all_predictions

if __name__ == '__main__':
    # leave_one_out_Test = True
    # if leave_one_out_Test:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--delta', type=str, default='0.2', help='delta value')
    args = parser.parse_args()
    # else
    base_dir = '/data2/user2/senkang/CP-GuardBench/CP-GuardBench_RawData'
    # pretrained_weights_path = '/data2/user2/senkang/CP-GuardBench/cpguard/logs/2024-11-30-18-42-51_CADet/10.pth'
    pretrained_weights_path = '/data2/user2/senkang/CP-GuardBench/cpguard/logs/2024-11-28-19-05-06_msc_loss/49.pth'

    eval_dirs = [
        'test_pgd_',
        'test_GN_',
        'test_fgsm_',
        'test_cw-l2_',
        'test_bim_'
    ]
    # eval_dirs = ['test']
    delta = args.delta

    overall_results = {}
    all_labels = []
    all_predictions = []

    import time

    total_time = 0
    total_samples = 0

    for eval_dir in eval_dirs:
        full_path = os.path.join(base_dir, eval_dir+str(delta))
        start_time = time.time()
        
        results, labels, predictions = test_model(full_path, pretrained_weights_path)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        total_samples += len(labels)
        
        overall_results[eval_dir] = results
        all_labels.extend(labels)
        all_predictions.extend(predictions)

        # Calculate and print FPS for this iteration
        fps = len(labels) / elapsed_time
        print(f"FPS for {eval_dir}: {fps:.2f}")

    # Calculate and print average FPS across all iterations
    avg_fps = total_samples / total_time
    print(f"Average FPS across all tests: {avg_fps:.2f}")

    print("Overall Results:")
    for eval_dir, results in overall_results.items():
        print(f"{eval_dir}_{delta}:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
        print()

    # Calculate and print average results
    avg_results = {metric: sum(results[metric] for results in overall_results.values()) / len(overall_results)
                   for metric in overall_results[eval_dirs[0]].keys()}
    
    print("Average Results:")
    for metric, value in avg_results.items():
        print(f"{metric}: {value:.4f}")

    # Plot and save overall ROC curve
    plot_roc_curve(all_labels, all_predictions, save_path='roc_curve/overall_roc_'+str(delta)+'_msc.png')
