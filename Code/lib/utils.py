import numpy as np
import torch
import os
from tslearn.clustering import TimeSeriesKMeans, KShape

def common_loss2(adj_1, adj_2):
    adj_1 = adj_1 - torch.eye(adj_1.shape[0]).cuda()
    adj_2 = torch.eye(adj_2.shape[0]).cuda() - adj_2
    cost = torch.sum((adj_1 - adj_2) ** 2)
    cost = torch.exp(-cost)
    return cost


def common_loss(adj_1, adj_2):
    adj_1 = adj_1 * (1 - torch.eye(adj_1.shape[0]).cuda())
    adj_2 = torch.eye(adj_2.shape[0]).cuda() - adj_2
    cost = torch.sum((adj_1 - adj_2) ** 2)
    cost = torch.exp(-cost)
    return cost

def dependence_loss(adj_1, adj_2):
    node_num = adj_1.shape[0]
    R = torch.eye(node_num) - (1/node_num) * torch.ones(node_num, node_num)
    adj_1 = adj_1 * (1 - torch.eye(adj_1.shape[0]).cuda())
    adj_2 = torch.eye(adj_2.shape[0]).cuda() - adj_2
    K1 = torch.mm(adj_1, adj_1.T)
    K2 = torch.mm(adj_2, adj_2.T)
    RK1 = torch.mm(R.cuda(), K1)
    RK2 = torch.mm(R.cuda(), K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def loss_dependence2(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def sample_subsequences(array, max_t):
    """
    对于一个 (T, N) 的 numpy 数组，生成一个包含多个子序列数组的列表，每个数组具有不同的子序列长度。

    参数:
    array (np.ndarray): 形状为 (T, N) 的输入数组。
    max_t (int): 需要从每个时间步采样的序列长度 t。

    返回:
    list[np.ndarray]: 一个包含不同长度的子序列数组的列表。
    """
    T, N = array.shape
    subsequences = []

    for i in range(T + 1 - max_t):
        # 对于每个 t，生成子序列
        subsequence = array[i:i + max_t]
        subsequences.append(subsequence)

    return np.stack(subsequences, axis=0)


def load_patterns(dataset):

    if dataset == 'PEMSD4':
        data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data_path = os.path.join('../data/PeMSD7/pems07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD3':
        data_path = os.path.join('../data/PeMSD3/pems03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'KnowAir':
        data_path = os.path.join('../data/KnowAir/KnowAir.npy')
        data = np.load(data_path)[:, :, 0]  #  column 3: Temperature  column -1: Pm2.5

    mean = data.mean()
    std = data.std()

    data = ((data - mean) / std)[:14*288,:]

    data_pattern = sample_subsequences(data, 3)
    data_pattern = data_pattern.swapaxes(1, 2)
    data_pattern = data_pattern.reshape(-1, 3, 1)

    if not os.path.exists(f'../data/{dataset}/{dataset}_pattern.npy'):
        km = KShape(n_clusters=12, max_iter=5).fit(data_pattern)
        centroid_patterns = km.cluster_centers_
        np.save(f'../data/{dataset}/{dataset}_pattern.npy', centroid_patterns)

    centroid_patterns = np.load(f'../data/{dataset}/{dataset}_pattern.npy')

    return centroid_patterns