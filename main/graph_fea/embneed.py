
import sys
import warnings

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from evaluate_embedding import evaluate_embedding
from gin import Encoder
from losses import local_global_loss_
from model import FF, PriorDiscriminator
from torch import optim
from torch.autograd import Variable
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import json
import json
import numpy as np
import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT_PATH = '/root/autodl-tmp/no-codingRNA-pretrain/main'
device = 'cuda:0'
sys.path.append(ROOT_PATH)
from arguments import arg_parse
import info_dataset as rna_data
args = arg_parse()

# def set_seed(args):
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)


class InfoGraph(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(InfoGraph, self).__init__()
        # self.dataset = rna_data.choose_dataset(args.key)
        # dataset_num_features = max(self.dataset.num_features, 1)
        # print(dataset_num_features, 'dataset_num_features')
        self.encoder = Encoder(3, 32, 3)
        # self.dataset = rna_data.dataset

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior
        self.embedding_dim = mi_units = hidden_dim * num_gc_layers


        # 下面都是计算损失函数的
        # 这里是全局特征和局部特征
        # self.local_d = FF(self.embedding_dim)
        # self.global_d = FF(self.embedding_dim)

        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self):
        pass


    def a(self, current_index, infoGraph, key):
        # set_seed(args)
        print(current_index, '我是current_index')
        # 这段代码是为了找到每个batch_Size
        star_index = current_index
        end_index = current_index + args.per_gpu_train_batch_size
        dataset = rna_data.choose_dataset(key)
        print(len(dataset), '我是图中的数据的长度')
        if end_index > len(dataset):
            dataset = dataset[current_index:]
        else:
            dataset = dataset[star_index:end_index]
        # 一直到这段
        dataloader = DataLoader(dataset, batch_size=args.per_gpu_train_batch_size)
        emb, y = infoGraph.encoder.get_embeddings(dataloader)
        # 在融合之前先添加一个全连接层，然后进行线性映射，为了让他与bert维度相同
        mapping_matrix = np.random.rand(96, 768)
        emb_map = emb.dot(mapping_matrix)
        # print(emb_map.shape, '我是最后的维度')
        return emb_map
def initialize_optimizer():
    model = InfoGraph(32, 3).to(device)
    model_dict = model.state_dict()
    if os.path.exists(args.infoGraph_model):
        checkpoint = torch.load(args.infoGraph_model)
        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    else:
        pass
    return model








