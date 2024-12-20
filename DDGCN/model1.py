from ctypes import sizeof
import torch.nn as nn
import torch.nn.functional as F
# from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
from torchstat import stat
import math
from torch.nn.modules.module import Module
# from config import Config
import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import numpy as np
import networkx as nx
from thop import profile

def common_loss(emb1, emb2):# consistency constraint
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    return cost

def loss_dependence(emb1, emb2, dim): #disparity constraint
    R = torch.eye(dim) - (1/dim) * torch.ones(dim, dim)
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):#把一个sparse matrix 转为torch稀疏张量
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    """
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):#标注训练集，测试集，验证集
    """Create mask."""
    '''
    e.g. Node IDs = [101,710,12,39,54,76,68,171,86,900]
    Train Mask = [True,True,True,True,False,False,False,False,False,False]
    Valodation Mask = [False,False,False,False,True,True,True,False,False,False]
    Test Mask = [False,False,False,False,False,False,False,True,True,True]
    '''
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) # 对每一行求和
    r_inv = np.power(rowsum, -1).flatten() # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    r_mat_inv = sp.diags(r_inv) # 构建对角元素为r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx

def load_data(config):
    f = np.loadtxt(config.feature_path, dtype = float)
    l = np.loadtxt(config.label_path, dtype = int)
    test = np.loadtxt(config.test_path, dtype = int)
    train = np.loadtxt(config.train_path, dtype = int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    idx_test = test.tolist()
    idx_train = train.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    label = torch.LongTensor(np.array(l))

    return features, label, idx_train, idx_test

def load_graph(dataset, config):#处理topology domain, feature domain
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    #直接从边表文件读取结果，(feature_edge_num,2)的数组，每一行表示一条边两个端点的idx
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    #将feature_edges中存储的是id,要将每一项的id变为 编号
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    #根据coo矩阵的性质，网络又多少条边，邻接矩阵就有多少个1，所以先建立一个长度为feature_edge_num的全1数据，每个1的填充位置就是一条边中两个端点的编号，
    #即fedges[:, 0], fedges[:, 1]
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    #build symmetric adjacency matrix  A^=(D~)^0.5 A~ (D~)^0.5
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))
    # eye创建单位矩阵，第一个参数为行数，第二个为列数, 对应公式A~=A+IN
    
    #基本操作同上
    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj+sp.eye(sadj.shape[0]))

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nsadj, nfadj



import configparser

class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))
        
        #Hyper-parameter
        self.epochs = conf.getint("Model_Setup", "epochs")
        self.lr = conf.getfloat("Model_Setup", "lr")
        self.weight_decay = conf.getfloat("Model_Setup", "weight_decay")
        self.k = conf.getint("Model_Setup", "k")
        self.nhid1 = conf.getint("Model_Setup", "nhid1")
        self.nhid2 = conf.getint("Model_Setup", "nhid2")
        self.dropout = conf.getfloat("Model_Setup", "dropout")
        self.beta = conf.getfloat("Model_Setup", "beta")
        self.theta = conf.getfloat("Model_Setup", "theta")
        self.no_cuda = conf.getboolean("Model_Setup", "no_cuda")
        self.no_seed = conf.getboolean("Model_Setup", "no_seed")
        self.seed = conf.getint("Model_Setup", "seed")

        # Dataset
        self.n = conf.getint("Data_Setting", "n")
        self.fdim = conf.getint("Data_Setting", "fdim")
        self.class_num = conf.getint("Data_Setting", "class_num")
        self.structgraph_path = conf.get("Data_Setting", "structgraph_path")
        self.featuregraph_path = conf.get("Data_Setting", "featuregraph_path")
        self.feature_path = conf.get("Data_Setting", "feature_path")
        self.label_path = conf.get("Data_Setting", "label_path")
        self.test_path = conf.get("Data_Setting", "test_path")
        self.train_path = conf.get("Data_Setting", "train_path")

config_file = "./config/" + str(20) + str('citeseer') + ".ini"
config = Config(config_file)
sadj, fadj = load_graph(20, config)
features, labels, idx_train, idx_test = load_data(config)
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True): #初始化层：输入feature, 输出feature，权重，偏移
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight = torch.sparse.FloatTensor(in_features, out_features)
        '''
        常见用法self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))：
        首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter
        绑定到这个module里面，所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        '''
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            #Parameters 和 register_parameter都会向parameters写入参数，但后者可以支持字符串命名 
        # self.reset_parameters()

    def reset_parameters(self): #初始化权重
        stdv = 1. / math.sqrt(self.weight.size(1)) #size()函数主要用于统计矩阵元素的个数，后矩阵某一维上的元素的个数的函数 size(1)为行
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        '''
        前馈计算：A=x w(0)
        input X与权重W相乘，然后adj矩阵与他们的积稀疏乘
        直接输入与权重之间进行torch.mm操作，得到support，即XW
        support与adj进行torch.spmm操作，得到output，即AXW选择是否加bias 
        '''
        input = torch.sparse.FloatTensor(1000, self.in_features)
        support = torch.spmm(input, self.weight)
        # support = input*self.weight
        # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        output = torch.spmm(adj, support)
        # output = torch.addmm(adj, support)
        if self.bias is not None:
            return output.to_dense() + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self): #底层节点的参数，feature的个数；隐层节点个数；最终分类数;nfeat, nhid, out, dropout22,128,256,0.5
        super(GCN, self).__init__() #super().__init__()利用父类里的对象构造函数
        self.gc1 = GraphConvolution(22, 128) # gc1输入尺寸nfeat，输出尺寸nhid
        self.gc2 = GraphConvolution(128, 256) # gc2输入尺寸nhid，输出尺寸out
        self.dropout = 0.5
        self.attention = Attention(256)
    # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
        self.MLP = nn.Sequential(
            nn.Linear(256, 4),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        x = F.relu(self.gc1(x, sadj)) # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, sadj)
        # x, att = self.attention(x)
        output = self.MLP(x)
        return output
# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, out, dropout): #底层节点的参数，feature的个数；隐层节点个数；最终分类数
#         super(GCN, self).__init__() #super().__init__()利用父类里的对象构造函数
#         self.gc1 = GraphConvolution(nfeat, nhid) # gc1输入尺寸nfeat，输出尺寸nhid
#         self.gc2 = GraphConvolution(nhid, out) # gc2输入尺寸nhid，输出尺寸out
#         self.dropout = dropout
#     # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj)) # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
#         x = F.dropout(x, self.dropout, training = self.training)
#         x = self.gc2(x, adj)
#         return x

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
# 22, 4, 128,256,1000,0.5
# 
class SFGCN(nn.Module):
    def __init__(self):
        super(SFGCN, self).__init__()

        self.SGCN1 = GCN(22, 128,256, 0.5)
        self.SGCN2 = GCN(22, 128, 256, 0.5)
        self.CGCN = GCN(22, 128, 256, 0.5)

        self.dropout = 0.5
        self.a = nn.Parameter(torch.zeros(size=(256, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(256)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(256, 4),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph
        Xcom = (com1 + com2) / 2
        ##attention
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)
        return output, att, emb1, com1, com2, emb2, emb
if __name__ == '__main__':
    # model = SFGCN()
    model = GCN()
    input = torch.randn(1, 1, 1, 1)
    flops, params = profile(model, inputs=(input, ))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))