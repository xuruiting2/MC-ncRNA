# Optional: eliminating warnings
def warn(*args, **kwargs):
    pass

import sys
import warnings
warnings.warn = warn


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
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('/home/xuruiting/InfoGraph-master/RNA_pretrain/data_preprocessing')
sys.path.append('/home/xuruiting/ncRNA_ss/DNABERT-master')
from arguments import arg_parse
import info_dataset as rna_data
class InfoGraph(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(InfoGraph, self).__init__()
    print('我在info里面')
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
    # 这里是全局特征和局部特征
    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)
    # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
    # self.global_d = MIFCNet(self.embedding_dim, mi_units)

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

  def forward(self, x, edge_index, batch, num_graphs):
    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)

    # 全局特征
    g_enc = self.global_d(y)
    # 局部特征
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    # 这里是损失
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR
if __name__ == '__main__':
    args = arg_parse()
    # 四个验证性能的分类方法
    accuracies = {'logreg':[], 'svc':[], 'linearsvc':[], 'randomforest':[]}
    epochs = 1
    log_interval = 1  # 每隔多少轮就会出现一个日志
    batch_size = agrs.per_gpu_train_batch_size
    if os.path.exists(args.infoGraph-model):
        print(f"Path parameter exists: {args.infoGraph-model}")
        exit()
    # print(args, '我是args')
    # exit()
    lr = args.lr
    # DS = args.DS
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    # 这一步会先下载数据
    # dataset = TUDataset(path, name=DS).shuffle()
    # 这一步进行修改，换成我自己的数据
    dataset = rna_data.dataset
    print(len(dataset.data.y))
    # exit()
    # print(dataset[0])
    dataset_num_features = max(dataset.num_features, 1)
    # loader的长度就是迭代的次数，用总数据集的数量来除以你的batch，就是一个epoch要迭代的次数
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # print(len(dataloader))
    # print(dir(dataloader))
    # print(len(dataloader.dataset))
    # exit()

    print('================')
    print(dir(dataset))
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InfoGraph(args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    '''
    # 这一段代码是我自己加的
    model_path = '/home/xuruiting/ncRNA_ss/InfoGraph-master/unsupervised/trained_model.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    print(optimizer, '加载了权重文件的')
    # 以上的代码是我自己加的
    '''
    # 这是为了对比训练的性能，所以给出的测试的分类的方法，看看在训练之前的分类性能
    # emb, y = model.encoder.get_embeddings(dataloader)
    # print('===== Before training =====')
    # res = evaluate_embedding(emb, y)
    # accuracies['logreg'].append(res[0])
    # accuracies['svc'].append(res[1])
    # accuracies['linearsvc'].append(res[2])
    # accuracies['randomforest'].append(res[3])

    # 训练开始
    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for data in dataloader:
            # 这个data包含的是图的几个属性，我们一次会拿batch_size那么多的数据，可以通过y和id属性来看出来
            # print(data, 'data')
            # # 这里开始是为了验证是否有x为空，并且这里面的data是150个为一组
            # x_array = np.array(data.x)
            # has_nan = np.isnan(x_array).any()
            # print(has_nan, 'has_nan')

            data = data.to(device)
            # print(data.num_graphs, 'data.num_graphs')
            # print(len(data.y), 'data.y')
            # exit()
            # num_graphs表示的是图的个数
            print('我在loss上面应该是我先输出')
            loss = model(data.x, data.edge_index, data.batch, data.num_graphs)
            print('应该是我先输出吧')
            # 每一个图的损失乘以图的个数
            loss_all += loss.item() * data.num_graphs
            print('是在loss之后还是在之前')
            # print(loss.item(), 'lloss.item() ')
            loss.backward()
            optimizer.step()


        print('===== Epoch {}, Loss {} ====='.format(epoch, loss_all / len(dataloader)))


    #下面这些代码是为了验证一个epoch之后我们特征提取的性能，先注释掉
        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader)
            # print(emb.shape, 'emb.shape')
            # print(len(emb), 'len.emb')
            # x_array = np.array(emb)
            # has_nan = np.isnan(x_array).any()
            # print(has_nan, 'has_nan')
            # exit()
            res = evaluate_embedding(emb, y)
            accuracies['logreg'].append(res[0])
            accuracies['svc'].append(res[1])
            accuracies['linearsvc'].append(res[2])
            accuracies['randomforest'].append(res[3])
            print(accuracies)

    with open('unsupervised.log', 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s))
    #一直到这里都是这一段代码

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
checkpoint = torch.load('/home/xuruiting/ncRNA_ss/InfoGraph-master/unsupervised/trained_model.pth')
print(checkpoint, '我是checkpint')

'''
# 这个函数的作用是为了bert那边来调用的，这块是我自己写的
def getGraphFeature():
    print('运行了吗')
    exit()
    args = arg_parse()
    batch_size = 15
    lr = 0.001
    dataset = rna_data.dataset
    dataset_num_features = max(dataset.num_features, 1)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    # 读取之前训练的权重文件，然后看看是什么,这一段代码是我自己加上去的
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InfoGraph(args.hidden_dim, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(optimizer, '没加载权重文件的')
    # 读取.pth文件
    model_path = '/home/xuruiting/ncRNA_ss/InfoGraph-master/unsupervised/trained_model.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    print(optimizer, '加载了权重文件的')
    emb, y = model.encoder.get_embeddings(dataloader)
    # 在融合之前先添加一个全连接层，然后进行线性映射，为了让他与bert维度相同
    mapping_matrix = np.random.rand(96, 768)
    emb_map = emb.dot(mapping_matrix)
    print(emb_map.shape, '我是最后的维度')
'''




