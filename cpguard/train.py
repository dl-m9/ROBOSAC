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

class CPGuardDataset(Dataset):

    def __init__(self, data_dir, if_residual=True):
        self.data_dir = data_dir
        self.if_residual = if_residual
        self.data_dir_list = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir)]

    
    def __len__(self):
        return len(self.data_dir_list)
        


    def __getitem__(self, idx):
        
        current_file = self.data_dir_list[idx]
        with open(current_file, 'rb') as f:
            raw_data = pickle.load(f)


        zero_tensor = torch.zeros(256, 32, 32).to('cuda')


        if self.if_residual:
            residual_latent_feature = []
            ego_latent_feature = raw_data[1][0]
            for k, v in raw_data.items():
                if not torch.equal(v[0], zero_tensor) and k != 1:  # k == 1: ego agent
                    residual_latent_feature.append(
                        [ego_latent_feature - v[0], v[1]]
                    )
            return residual_latent_feature
        else:
            latent_feature = []
            for k, v in raw_data.items():
                if not torch.equal(v[0], zero_tensor) and k != 1:
                    latent_feature.append(
                        [v[0], v[1]]
                    )
            return latent_feature


import torch
import torch.nn as nn

# define the CNN



    

import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')  # He 初始化
        if m.bias is not None:
            init.constant_(m.bias, 0)

if __name__ == '__main__':
    data_dir = '/data2/user2/senkang/CP-GuardBench/CP-GuardBench_RawData/train'
    train_dataset = CPGuardDataset(data_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    eval_data_dir = '/data2/user2/senkang/CP-GuardBench/CP-GuardBench_RawData/test'
    eval_dataset = CPGuardDataset(eval_data_dir)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

    
    batch_size = 1 # do not change this 
    num_epochs = 100
    training = True
    resume = True


    if training:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()

        model = ResNetBinaryClassifier().to('cuda')
        # model.apply(init_weights).to('cuda')

        start_epoch = 0
        if resume:
            # Load the latest pretrained weights
            log_dir = '/data2/user2/senkang/CP-GuardBench/cpguard/logs/2024-09-02-20-32-18'
            pth_files = [f for f in os.listdir(log_dir) if f.endswith('.pth')]
            latest_pth = max(pth_files, key=lambda x: int(x.split('.')[0]))
            pretrained_weights_path = os.path.join(log_dir, latest_pth)
            
            model.load_state_dict(torch.load(pretrained_weights_path))
            start_epoch = int(latest_pth.split('.')[0]) + 1  # Start from the next epoch

        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

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
                for i, data in tqdm(enumerate(train_dataloader)):

                    optimizer.zero_grad()

                    residual_latent_feature_list = torch.stack([item[0] for item in data]).squeeze().to('cuda')
                    label_list = torch.tensor([item[1] for item in data], dtype=torch.float).to('cuda')

                    outputs = model(residual_latent_feature_list).squeeze()
                    # for output in outputs:
                    loss = criterion(outputs, label_list)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Progress bar
                    pbar.set_postfix(Epoch=f'{epoch+1}/{num_epochs}', loss=f'{loss.item():.4f}')
                    pbar.update(1)
                    
                    # Log loss to TensorBoard
                    writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + i)
            
            avg_loss = total_loss / len(train_dataloader)
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
                    for data in eval_dataloader:  # Changed train_dataloader to eval_dataloader
                        residual_latent_feature_list = torch.stack([item[0] for item in data]).squeeze().to('cuda')
                        label_list = torch.tensor([item[1] for item in data], dtype=torch.float).to('cuda')

                        outputs = model(residual_latent_feature_list).squeeze()
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

