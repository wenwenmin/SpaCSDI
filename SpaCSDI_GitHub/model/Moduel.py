import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)


def full_block(in_features, out_features, act):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        act,
    )



class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.02, act=None, bn=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        # 使用 nn.ReLU() 而不是 F.relu
        self.act = act if act is not None else nn.ReLU()
        # 初始化 BatchNorm（可以选择是否启用）
        self.bn = nn.BatchNorm1d(out_features) if bn else nn.Identity()
        # GCN 卷积层
        self.conv = GCNConv(in_channels=self.in_features, out_channels=self.out_features)


    def forward(self, x, edge_index):
        device = x.device  # 以 x 的 device 为主

        # 处理 edge_index 和 edge_weight
        if isinstance(edge_index, torch.Tensor):
            edge_weight = None
        else:
            # 从 scipy sparse 转换
            coo = edge_index.tocoo()
            edge_index = torch.tensor(
                np.vstack([coo.row, coo.col]),
                dtype=torch.long,
                device=device
            )
            edge_weight = (
                torch.tensor(coo.data, dtype=torch.float32, device=device)
                if coo.data is not None
                else None
            )
            if edge_weight is None:
                print("edge_weight没有值！！！")

        # 确保 edge_weight 存在
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)

        # Graph Convolution
        x = self.conv(x, edge_index, edge_weight)
        x = self.bn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act(x)
        return x


"""基于距离的编码器"""


class DisGraphConv(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.01, act=nn.ReLU(), bn=True):
        super(DisGraphConv, self).__init__()
        self.in_features = in_features
        self.feat_hidden = hidden_features
        self.out_features = out_features
        self.layer1 = full_block(self.in_features, self.feat_hidden,act)
        # 第二层图卷积
        self.layer2 = GraphConv(self.feat_hidden, self.out_features, dropout=dropout, act=act, bn=bn)

    @torch.no_grad()
    def inference(self, x, edge_index):
        x = self.layer1(x)
        x = self.layer2(x, edge_index)
        return x

    def forward(self, x, edge_index):
        x = self.layer1(x)
        x = self.layer2(x, edge_index)
        return x


"""基于基因表达矩阵相似度的编码器"""


class ExpGraphConv(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.01, act=nn.ReLU(), bn=True):
        super(ExpGraphConv, self).__init__()
        self.in_features = in_features
        self.feat_hidden = hidden_features
        self.out_features = out_features
        self.layer1 = full_block(self.in_features, self.feat_hidden,act)
        # 第二层图卷积
        self.layer2 = GraphConv(self.feat_hidden, self.out_features, dropout=dropout, act=act, bn=bn)

    @torch.no_grad()
    def inference(self, x, edge_index):
        # 第一层
        x = self.layer1(x)
        # 第二层
        x = self.layer2(x, edge_index)
        return x

    def forward(self, x, edge_index):
        # 第一层
        x = self.layer1(x)
        # 第二层
        x = self.layer2(x, edge_index)
        return x


"""解码器"""


class Decoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.001, act=nn.ReLU(), bn=True):
        super(Decoder, self).__init__()
        self.in_features = in_features
        self.feat_hidden = hidden_features
        self.out_features = out_features
        self.layer1 = full_block(self.in_features, self.feat_hidden,act)
        self.layer2 = GraphConv(self.feat_hidden, self.feat_hidden, dropout=dropout, act=act, bn=bn)
        self.layer3 = full_block(self.feat_hidden, 2*self.feat_hidden,act=act)
        self.layer4 = full_block(2*self.feat_hidden, self.out_features, act=act)


    def forward(self, x, edge_index, x_true, mask):
        # 确保输入的 mask 是布尔类型
        # assert mask.dtype == torch.bool, "Mask should be of boolean dtype!"
        # 第一层
        x_1 = self.layer1(x)
        # 第二层
        x_2 = self.layer2(x_1, edge_index)
        x_3=self.layer3(x_2)
        x=self.layer4(x_3)
        if mask !=None:
             mask=mask.to(x.device )
             assert x.shape[0] == mask.shape[0], f"Shape mismatch: x.shape[0] = {x.shape[0]}, mask.shape[0] = {mask.shape[0]}"
             assert x.shape == x_true.shape, f"Shape mismatch: x.shape = {x.shape}, x_true.shape = {x_true.shape}"
             ren_loss = nn.MSELoss()(x * (~mask), x_true * (~mask))
             return ren_loss
        else:
            ren_loss = nn.MSELoss()(x , x_true)
            return ren_loss

    @torch.no_grad()
    def inference(self, x, edge_index):
        x_1 = self.layer1(x)
        # 第二层
        x_2 = self.layer2(x_1, edge_index)
        x_3 = self.layer3(x_2)
        x = self.layer4(x_3)
        return x




"""判别器"""

#针对去批次的判别器
class Discriminator_batch(nn.Module):
    def __init__(self, input_dim, latent_dim, batch_num, p_drop=0.01,act=nn.LeakyReLU()):
        super(Discriminator_batch, self).__init__()
        self.discriminator = nn.Sequential(
            full_block(input_dim, latent_dim, act),
            full_block(latent_dim, int(latent_dim / 2), act),
            nn.Linear(int(latent_dim / 2), batch_num)
        )

    @torch.no_grad()
    def evaluate(self, x, label):
        logits = self.discriminator(x)  # 这里是 logits 不是 softmax 结果
        pred = F.softmax(logits, dim=1)

        if isinstance(label, pd.DataFrame):
            label = torch.tensor(label.values, dtype=torch.long).to(x.device)  # 转换为 long 类型

        loss = F.cross_entropy(logits, label)  # 使用 logits 计算 loss
        pred_class = pred.argmax(dim=1)
        accuracy = (pred_class == label).float().mean()

        return accuracy, loss

    def forward(self, x, label):
        logits = self.discriminator(x)
        if isinstance(label, pd.DataFrame):
            label = torch.tensor(label.values, dtype=torch.long).to(x.device)  # 转换为 long 类型
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label.values, dtype=torch.long)  # 转换成 Tensor
        dis_loss = F.cross_entropy(logits, label)  # 修正 loss 计算
        return dis_loss

#针对domain识别的判别器
class Discriminator_domain(nn.Module):
    def __init__(self, input_dim, latent_dim, class_num, p_drop=0.01,act=nn.LeakyReLU()):
        super(Discriminator_domain, self).__init__()
        self.discriminator = nn.Sequential(
            full_block(input_dim, latent_dim, act),
            full_block(latent_dim, int(latent_dim / 2), act),
            nn.Linear(int(latent_dim / 2), class_num)
        )

    def forward(self, x, label):
        logits = self.discriminator(x)
        if isinstance(label, pd.DataFrame):
            label = torch.tensor(label.values, dtype=torch.long).to(x.device)  # 转换为 long 类型
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label.values, dtype=torch.long)  # 转换成 Tensor
        dis_loss = F.cross_entropy(logits, label)  # 修正 loss 计算
        return dis_loss

class Predictor(nn.Module):
    def __init__(self, input_num, latent_dim, class_num, p_drop=0.02,act=nn.LeakyReLU()):
        super(Predictor, self).__init__()
        self.input_num = input_num
        self.class_num = class_num
        self.latent_dim = latent_dim
        self.p_drop = p_drop
        self.latent_dim1 = int(latent_dim / 4)

        self.predictor = nn.Sequential(
            full_block(self.input_num, self.latent_dim,act),
            full_block(self.latent_dim, self.latent_dim1, act),
            nn.Linear(self.latent_dim1, self.class_num),
            nn.Softmax(dim=1)
        )

    @torch.no_grad()
    def predicted(self, x):
        x = self.predictor(x)
        return x

    def forward(self, x, label):
        x = self.predictor(x)
        if isinstance(label, pd.DataFrame):
            label = torch.tensor(label.values, dtype=torch.float32).to(x.device)
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).float().to(x.device)
        loss = nn.MSELoss()(x, label)
        loss1 = js_divergence(label, x)
        rec_dis=loss1+loss
        return x, rec_dis


def js_divergence(p, q, eps=1e-10):
    p = p + eps
    q = q + eps
    m = 0.5 * (p + q)  # 计算均值分布
    return 0.5 * (F.kl_div(torch.log(p), m, reduction='batchmean') +
                  F.kl_div(torch.log(q), m, reduction='batchmean'))


