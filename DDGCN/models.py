import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
from torchstat import stat
import math

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout): #底层节点的参数，feature的个数；隐层节点个数；最终分类数
        super(GCN, self).__init__() #super().__init__()利用父类里的对象构造函数
        self.gc1 = GraphConvolution(nfeat, nhid) # gc1输入尺寸nfeat，输出尺寸nhid
        self.gc2 = GraphConvolution(nhid, out) # gc2输入尺寸nhid，输出尺寸out
        self.dropout = dropout
    # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj)) # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x


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
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(SFGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj):
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
# if __name__ == '__main__':
#     model = SFGCN(22, 4, 128,256,1000,0.5)
#     stat(model,(128,256,256))