import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pickle
from tqdm import tqdm
import time
from sklearn.metrics import precision_score, recall_score, f1_score
from models import ResNetBinaryClassifier, detector
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.init as init
from pytorch_metric_learning import losses



class CPGuardDataset(Dataset):

    def __init__(self, data_dir, if_residual=True):
        self.data_dir = data_dir
        self.if_residual = if_residual
        self.data_dir_list = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if file.endswith('.pkl')]

    
    def __len__(self):
        return len(self.data_dir_list)
        


    def __getitem__(self, idx):
        
        current_file = self.data_dir_list[idx]
        with open(current_file, 'rb') as f:
            raw_data = pickle.load(f)


        zero_tensor = torch.zeros(256, 32, 32).to('cuda')
        # for k, v in raw_data.items(): # 过滤掉所有值为0的元素
        #     if torch.equal(v[0], zero_tensor):
        if self.if_residual:
            residual_latent_feature = []
            ego_latent_feature = raw_data[1][0]
            for k, v in raw_data.items():
                if not torch.equal(v[0], zero_tensor) and k != 1:  # k == 1: ego agent
                    residual = ego_latent_feature - v[0]
                    # Normalize the residual
                    residual_norm = (residual - residual.mean()) / (residual.std() + 1e-8)
                    residual_latent_feature.append([residual_norm, v[1]])


            return residual_latent_feature, current_file.split('/')[-1]
        else:
            latent_feature = []
            for k, v in raw_data.items():
                if not torch.equal(v[0], zero_tensor) and k != 1:
                    latent_feature.append(
                        [v[0], v[1]]
                    )
            return latent_feature, current_file.split('/')[-1]


# define the CNN






    


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')  # He 初始化
        if m.bias is not None:
            init.constant_(m.bias, 0)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size (do not change this)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')

    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    training = True
    resume = args.resume
    if_residual = False
    if_cont_training = False


    data_dir = '/data2/user2/senkang/CP-GuardBench/CP-GuardBench_RawData/generated'
    train_dataset = CPGuardDataset(data_dir, if_residual=if_residual)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    eval_data_dir = '/data2/user2/senkang/CP-GuardBench/CP-GuardBench_RawData/test'
    eval_dataset = CPGuardDataset(eval_data_dir, if_residual=if_residual)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    
    


    if training:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

        model = ResNetBinaryClassifier().to('cuda')
        cont_loss = losses.NTXentLoss()

        start_epoch = 0
        if resume:
            # Load the latest pretrained weights
            log_dir = '/data2/user2/senkang/CP-GuardBench/cpguard/logs/2024-09-07-19-18-38'
            pth_files = [f for f in os.listdir(log_dir) if f.endswith('.pth')]
            latest_pth = max(pth_files, key=lambda x: int(x.split('.')[0]))
            pretrained_weights_path = os.path.join(log_dir, latest_pth)
            
            model.load_state_dict(torch.load(pretrained_weights_path))
            start_epoch = int(latest_pth.split('.')[0]) + 1  # Start from the next epoch

        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)

        # config log path
        if not resume:
            log_path = '/data2/user2/senkang/CP-GuardBench/cpguard/logs'
            current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            log_dir = log_path+f'/{current_time}'
            os.makedirs(log_path+f'/{current_time}', exist_ok=True)
        
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss = 0.0
            with tqdm(total=len(train_dataloader)) as pbar:
                batch_data = []
                batch_label = [] 
                for i, (data, file_name) in tqdm(enumerate(train_dataloader)):
                    # residual_latent_feature_list = []
                    # label_list = []
                    # for j in range(batch_size):
                    #     residual_latent_feature_list.extend(data[0])
                    #     label_list.extend(data[1])

                    # optimizer.zero_grad()
                    # if file_name[0] != '92_0.pkl': continue
                    if len(data) == 0: continue

                    
                    
                    residual_latent_feature_list = torch.cat([item[0] for item in data]).to('cuda')
                    
                    # Check if residual_latent_feature_list is 4D, if not, add a dimension
                    # if residual_latent_feature_list.dim() != 4:
                    #     residual_latent_feature_list = residual_latent_feature_list.unsqueeze(0)
                    labels = torch.cat([item[1].to(dtype=torch.float, device='cuda') for item in data], dim=0)

                    batch_data.append(residual_latent_feature_list)
                    batch_label.append(labels)

                    if (i + 1) % batch_size == 0 or i == len(train_dataloader) - 1:
                        residual_latent_feature_list = torch.cat(batch_data, dim=0)
                        labels = torch.cat(batch_label, dim=0)

                        optimizer.zero_grad()

                        outputs, embeddings = model(residual_latent_feature_list)
                        outputs = outputs.squeeze()
                        loss = criterion(outputs, labels)
                        if if_cont_training:
                            cont_loss_value = cont_loss(embeddings, labels)
                            loss = loss + cont_loss_value * 0.1

                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        
                        # Progress bar
                        pbar.set_postfix(Epoch=f'{epoch+1}/{num_epochs}', loss=f'{loss.item():.4f}')
                        pbar.update(1)
                        
                        # Log loss to TensorBoard
                        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + i)

                        batch_data = []
                        batch_label = []
            
            avg_loss = total_loss / (len(train_dataloader) / batch_size)
            print(f'Average Loss for Epoch {epoch+1}: {avg_loss:.4f}')
            torch.save(model.state_dict(), f'{log_dir}/{epoch}.pth')
            
            # Log average loss and epoch to TensorBoard
            writer.add_scalar('Loss/avg_train', avg_loss, epoch)
            
            

                
            

            model.eval()
            eval_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                with tqdm(total=len(eval_dataloader), desc="Evaluating") as pbar:  # Changed train_dataloader to eval_dataloader
                    for (data, file_name) in eval_dataloader:  # Changed train_dataloader to eval_dataloader
                        residual_latent_feature_list = torch.stack([item[0] for item in data]).squeeze().to('cuda')
                        label_list = torch.tensor([item[1] for item in data], dtype=torch.float).to('cuda')

                        outputs, _ = model(residual_latent_feature_list)
                        outputs = outputs.squeeze() 
                        loss = criterion(outputs, label_list)
                        eval_loss += loss.item()

                        predicted = (outputs > 0.5).float()
                        
                        # print(outputs, predicted)
                        correct_predictions += (predicted == label_list).sum().item()
                        total_predictions += label_list.size(0)

                        all_labels.extend(label_list.cpu().numpy())
                        all_predictions.extend(predicted.cpu().numpy())

                        # Progress bar
                        pbar.update(1)

            avg_eval_loss = eval_loss / len(eval_dataloader)  # Changed train_dataloader to eval_dataloader
            accuracy = correct_predictions / total_predictions
            precision = precision_score(all_labels, all_predictions)
            recall = recall_score(all_labels, all_predictions)
            f1 = f1_score(all_labels, all_predictions)

            print(f'Evaluation Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
            
            # Log evaluation metrics to TensorBoard
            writer.add_scalar('Loss/eval', avg_eval_loss, epoch)
            writer.add_scalar('Accuracy/eval', accuracy, epoch)
            writer.add_scalar('Precision/eval', precision, epoch)
            writer.add_scalar('Recall/eval', recall, epoch)
            writer.add_scalar('F1-Score/eval', f1, epoch)

            writer.flush()  # Ensure all pending events have been written to disk

