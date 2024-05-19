import torch
import torch.nn as nn
import torch.nn.functional as F
from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation

def local_global_loss_(l_enc, g_enc, edge_index, batch, measure):
    '''
    Args:
        l: Local feature map.代表了局部特征
        g: Global features.代表了全局特征
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''

    # 一共有batch_size那么多的图片，那么多的图片的节点数据是多少
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]
    # print(num_graphs, 'num_graphs')
    # print(num_nodes, 'num_nodes')
    # exit()
    # 创建了两个张量，第一个张量的所有初始化都是0，第二个张量的所有的元素的初始化都是1
    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos

def adj_loss_(l_enc, g_enc, edge_index, batch):
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    adj = torch.zeros((num_nodes, num_nodes)).cuda()
    mask = torch.eye(num_nodes).cuda()
    for node1, node2 in zip(edge_index[0], edge_index[1]):
        adj[node1.item()][node2.item()] = 1.
        adj[node2.item()][node1.item()] = 1.

    res = torch.sigmoid((torch.mm(l_enc, l_enc.t())))
    res = (1-mask) * res
    # print(res.shape, adj.shape)
    # input()

    loss = nn.BCELoss()(res, adj)
    return loss
