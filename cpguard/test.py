
import torch
from torch.utils.data import DataLoader
import os
from train import CPGuardDataset, detector, init_weights
from sklearn.metrics import precision_score, recall_score, f1_score
from models import ResNetBinaryClassifier

def test_model():
    eval_data_dir = '/data2/user2/senkang/CP-GuardBench/CP-GuardBench_RawData/train'
    eval_dataset = CPGuardDataset(eval_data_dir)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

    model = ResNetBinaryClassifier().to('cuda')
    pretrained_weights_path = '/data2/user2/senkang/CP-GuardBench/cpguard/logs/resnet50-2024-09-02-20-32-18/299.pth'
    model.load_state_dict(torch.load(pretrained_weights_path))
    model.to('cuda')
    model.eval()

    criterion = torch.nn.BCELoss()

    eval_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i, data in enumerate(eval_dataloader):
            residual_latent_feature_list = torch.stack([item[0] for item in data]).squeeze().to('cuda')
            label_list = torch.tensor([item[1] for item in data], dtype=torch.float).to('cuda')

            outputs = model(residual_latent_feature_list).squeeze()
            loss = criterion(outputs, label_list)
            eval_loss += loss.item()

            predicted = (outputs > 0.01).float()
            correct_predictions += (predicted == label_list).sum().item()
            # print(i, correct_predictions)
            total_predictions += label_list.size(0)

            all_labels.extend(label_list.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_eval_loss = eval_loss / len(eval_dataloader)
    accuracy = correct_predictions / total_predictions
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f'Evaluation Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

if __name__ == '__main__':
    test_model()
