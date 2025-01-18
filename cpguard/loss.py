import torch
import torch.nn.functional as F

def MCL_loss(embeddings, labels, alpha=0.05, t=0.2):
    """
    Class Conditional Masking (CCM) loss function for contrastive learning
    
    参数说明:
    embeddings: 输入特征向量 (batch_size x 2048)
    labels: 标签向量 (batch_size)
    alpha: 控制同类样本之间的吸引力度的系数,默认为0.05  
    t: temperature参数,用于调节相似度分数的scale,默认为0.2
    
    实现步骤:
    1. 对输入特征进行L2归一化
    """
    embeddings = F.normalize(embeddings, dim=1)
    
    """
    2. 构建mask矩阵:
    - 复制标签向量y形成矩阵c
    - 每行减去对角线元素,得到样本间的标签差异矩阵mask
    """
    c = [labels]*len(labels)
    mask = torch.stack(c).to(embeddings.device)  # Move to same device as embeddings
    for i in range(len(mask)):
        mask[i] = mask[i]- mask[i][i]
    
    """
    3. 根据mask矩阵构建不同的mask:
    - mask_pos: 标签差异为正的位置 
    - mask_neg: 标签差异为负的位置
    - mask_eq: 完全相同类别的位置(差值为0)
    - diag: 对角线位置
    """
    mask_pos = (mask>0).type(torch.float).to(embeddings.device)  /t  # 正差异位置
    mask_neg = (mask<0).type(torch.float).to(embeddings.device)  /t  # 负差异位置
    mask_eq = (mask==0).type(torch.float).to(embeddings.device) * alpha   # 类别相同位置
    diag = torch.eye(len(labels), device=embeddings.device) * (1/t - alpha)  # 对角线位置
    
    """
    4. 合并所有mask并进行维度调整:
    - 将各个mask相加得到mask_sum
    - 调整维度并上采样得到最终的mask_final
    """
    mask_sum = mask_pos + mask_neg + diag + mask_eq
    mask_sum = mask_sum.view(1,1,len(labels),len(labels))
    # mask_final = F.interpolate(mask_sum,scale_factor = 1,mode='nearest')
    mask_final = mask_sum.squeeze()

    """
    5. 计算样本间的余弦相似度
    """
    x_scores =  (embeddings @ embeddings.t()).clamp(min=1e-7)  # 计算归一化的余弦相似度

    """
    6. 应用mask到相似度矩阵
    """
    x_scale = x_scores * mask_final

    """
    7. 准备交叉熵损失的计算:
    - 将对角线置为很大的负值(softmax后接近0)
    - 构建目标向量,相邻位置互为正样本
    """
    x_scale = x_scale - torch.eye(x_scale.size(0), device=embeddings.device) * 1e5
    targets = torch.arange(embeddings.size()[0], device=embeddings.device)
    targets[::2] += 1  # 偶数位置+1
    targets[1::2] -= 1  # 奇数位置-1
    
    """
    8. 返回交叉熵损失
    """
    return F.cross_entropy(x_scale, targets.long())


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLossWithMMD(nn.Module):
    """
    基于 MMD (Maximum Mean Discrepancy) 的 NT-Xent Loss
    """

    def __init__(self, temperature=0.5, kernel="rbf", gamma=None):
        """
        初始化 NT-Xent 损失函数

        参数:
            temperature: float，温度参数 τ，用于缩放相似度。
            kernel: str，核函数类型，可选 'rbf' 或 'linear'。
            gamma: float，RBF 核的超参数 (默认 None)。
        """
        super(NTXentLossWithMMD, self).__init__()
        self.temperature = temperature
        self.kernel = kernel
        self.gamma = gamma

    def forward(self, embeddings, labels):
        """
        计算 NT-Xent 损失

        参数:
            embeddings: Tensor，形状为 (batch_size, embedding_dim)，嵌入向量。
            labels: Tensor，形状为 (batch_size,)，样本的标签。

        返回:
            loss: 标量张量，表示对比损失
        """
        # 归一化嵌入向量 (L2 normalization)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 计算 MMD 相似度矩阵
        similarity_matrix = compute_mmd(embeddings, embeddings, kernel=self.kernel, gamma=self.gamma)

        # 缩放相似度矩阵
        similarity_matrix = similarity_matrix / self.temperature

        # 构造标签掩码 (正样本对和负样本对)
        labels = labels.view(-1, 1)  # 将 labels 转换为 (batch_size, 1)
        mask = torch.eq(labels, labels.T).float()  # (batch_size, batch_size)，正样本对为 1，负样本对为 0

        # 对角线排除自身相似性 (避免样本和自身作为正样本)
        logits = similarity_matrix - torch.eye(similarity_matrix.size(0), device=similarity_matrix.device) * 1e9

        # 计算 softmax 分布
        exp_logits = torch.exp(logits)  # 对相似度取 exp
        exp_logits_sum = exp_logits.sum(dim=1, keepdim=True)  # 按行求和

        # 正样本对的相似性
        positive_logits = torch.exp(similarity_matrix) * mask  # 仅保留正样本对
        positive_logits_sum = positive_logits.sum(dim=1)

        # 计算对比损失
        loss = -torch.log(positive_logits_sum / exp_logits_sum + 1e-8).mean()  # 加 1e-8 防止 log(0)

        return loss

def compute_mmd(x, y, kernel="rbf", gamma=None):
    """
    计算两个样本之间的 MMD 相似度。

    参数:
        x: Tensor，样本集合 1，形状为 (n_samples_x, embedding_dim)。
        y: Tensor，样本集合 2，形状为 (n_samples_y, embedding_dim)。
        kernel: str，核函数类型，可选 'rbf' 或 'linear' (默认 'rbf')。
        gamma: float，RBF 核的超参数 (如果未指定，默认为 1 / embedding_dim)。

    返回:
        mmd_value: Tensor，MMD 值，形状为 (n_samples_x, n_samples_y)
    """
    if kernel == "rbf":
        if gamma is None:
            gamma = 1.0 / x.size(1)  # 默认 gamma = 1 / 特征维度
        xx = torch.sum(x ** 2, dim=1, keepdim=True)  # 样本 x 的平方和
        yy = torch.sum(y ** 2, dim=1, keepdim=True)  # 样本 y 的平方和
        xy = torch.matmul(x, y.T)  # 样本间的点积
        distances = xx + yy.T - 2 * xy  # 欧氏距离的平方
        kernel_matrix = torch.exp(-gamma * distances)  # RBF 核
    elif kernel == "linear":
        kernel_matrix = torch.matmul(x, y.T)  # 线性核
    else:
        raise ValueError(f"Unsupported kernel type: {kernel}")

    return kernel_matrix
