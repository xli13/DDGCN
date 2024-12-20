import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True): #初始化层：输入feature, 输出feature，权重，偏移
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) #floatTensor建立tensor
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
        self.reset_parameters()

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
        support = torch.mm(input, self.weight)
        # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
