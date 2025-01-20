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
from loss import MCL_loss, NTXentLossWithMMD


class CPGuardDataset(Dataset):

    def __init__(self, data_dir, if_residual=True):
        self.data_dir = data_dir
        self.if_residual = if_residual
        self.data_dir_list = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if file.endswith('.pkl')]
        # self.data_dir_png_list = [file for file in os.listdir(self.data_dir) if file.endswith('.png')]
    
    def __len__(self):
        return len(self.data_dir_list)
        


    def __getitem__(self, idx):
        
        current_file = self.data_dir_list[idx]
        # # Get the base filename without extension
        # base_filename = os.path.splitext(os.path.basename(current_file))[0]
        # # Find matching png file that starts with the base filename
        # current_attack_file = [png_file for png_file in self.data_dir_png_list if png_file.startswith(base_filename)]
        # attack_list = [attack_type.split('_')[-1].split('.')[0] for attack_type in current_attack_file if attack_type.split('_')[-1].split('.')[0] is not 'ego']


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
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--leave_one_out', type=str, default='', help='Leave one out')
    parser.add_argument('--msc_loss', action='store_true', help='Use MSC loss')
    parser.add_argument('--mcl_loss', action='store_true', help='Use MCL loss')
    parser.add_argument('--CADet', action='store_true', help='Use CADet loss')
    parser.add_argument('--if_cont_training', action='store_true', help='Use contrastive training')
    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    training = True
    resume = args.resume
    if_residual = False

    if_cont_training = args.if_cont_training
    msc_loss = args.msc_loss
    mcl_loss = args.mcl_loss
    CADet_loss = True
    if CADet_loss:
        cadet_loss = NTXentLossWithMMD()
    

    leave_one_out = args.leave_one_out
    calculate_similarity = False

    data_dir = '/data2/user2/senkang/CP-GuardBench/CP-GuardBench_RawData/generated2'
    if leave_one_out:
        data_dir = '/data2/user2/senkang/CP-GuardBench/CP-GuardBench_RawData/generated2'
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
            log_dir = log_path+f'/{current_time}' + '_MSC_leave_one_out_' + leave_one_out
            os.makedirs(log_dir, exist_ok=True)

        if msc_loss:
            # 将模型设置为评估模式,不进行梯度计算和参数更新
            model.eval()
            
            # 初始化变量用于计算特征空间中心
            # embeddings_space_sum: 用于累加所有样本的特征向量
            # total_samples: 记录处理的总样本数
            embeddings_space_sum = None
            total_samples = 0
            
            # 使用torch.no_grad()避免计算和存储梯度
            with torch.no_grad():
                # 使用tqdm显示进度条
                with tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Calculating feature space center...') as pbar:
                    for i, (data, file_name) in pbar:
                        # 跳过空数据
                        if len(data) == 0: continue
                        
                        # 为了避免内存溢出,将数据分批处理
                        batch_size = 32
                        # 从data中提取特征列表
                        feature_list = [item[0] for item in data]
                        
                        # 按batch_size大小分批处理特征
                        for j in range(0, len(feature_list), batch_size):
                            # 将一批特征连接并移至GPU
                            batch_features = torch.cat(feature_list[j:j+batch_size]).to('cuda')
                            # 通过模型获取embeddings特征空间
                            _, embeddings_space = model(batch_features)
                            
                            # 累加特征向量
                            if embeddings_space_sum is None:
                                embeddings_space_sum = embeddings_space.sum(dim=0)
                            else:
                                embeddings_space_sum += embeddings_space.sum(dim=0)
                            # 更新样本总数
                            total_samples += embeddings_space.size(0)
                            
                            # 手动释放GPU内存
                            del batch_features, embeddings_space
                            torch.cuda.empty_cache()
                            
                        # 更新进度条
                        pbar.update(1)
                        
            # 计算特征空间的中心点:特征向量之和除以样本总数
            embeddings_space_center = embeddings_space_sum / total_samples

        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss = 0.0
            positive_similarities = []
            negative_similarities = []
            with tqdm(total=len(train_dataloader)) as pbar:
                batch_data = []
                batch_label = [] 
                for i, (data, file_name) in tqdm(enumerate(train_dataloader)):
                    if len(data) == 0: continue
                    

                    if leave_one_out:
                        residual_latent_feature_list = [item[0] for item in data if item[1][0] != leave_one_out]

                        if residual_latent_feature_list:  # Check if the list is not empty
                            residual_latent_feature_list = torch.cat(residual_latent_feature_list).to('cuda')
                            labels = torch.cat([torch.tensor([0.0] if item[1][0] == 'normal' else [1.0], dtype=torch.float, device='cuda') for item in data if item[1][0] != leave_one_out], dim=0)

                        else:
                            continue
        
                    else:
                        residual_latent_feature_list = torch.cat([item[0] for item in data]).to('cuda')
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

                            # 计算正样本对和负样本对的相似度
                            if calculate_similarity:
                                similarity_matrix = torch.matmul(embeddings, embeddings.t())
                                mask = labels.unsqueeze(0) == labels.unsqueeze(1)
                                positive_pairs = similarity_matrix[mask].view(-1)
                                negative_pairs = similarity_matrix[~mask].view(-1)

                            positive_similarities.extend(positive_pairs.detach().cpu().numpy())
                            negative_similarities.extend(negative_pairs.detach().cpu().numpy())
                        elif msc_loss:
                            cont_loss_value = cont_loss(embeddings-embeddings_space_center, labels)
                            loss = loss + cont_loss_value * 0.1
                        elif mcl_loss:
                            mcl_loss_value = MCL_loss(embeddings, labels, alpha=0.05, t=0.2)
                            loss = loss + mcl_loss_value * 0.1
                        elif CADet_loss:
                            cadet_loss_value = cadet_loss(embeddings, labels)
                            loss = loss + cadet_loss_value * 0.1

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
            
            # 记录正样本对和负样本对的平均相似度
            if positive_similarities:
                avg_positive_similarity = np.mean(positive_similarities)
                writer.add_scalar('Similarity/positive', avg_positive_similarity, epoch)
            if negative_similarities:
                avg_negative_similarity = np.mean(negative_similarities)
                writer.add_scalar('Similarity/negative', avg_negative_similarity, epoch)
            

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
