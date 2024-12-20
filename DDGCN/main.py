from __future__ import division
from __future__ import print_function
import importlib
from statistics import mode
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import Attention
from utils import *
from models import SFGCN
import numpy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn import manifold
import os
import torch.nn as nn
import argparse
from config import Config
import matplotlib.pyplot as plt
from visdom import Visdom
from openTSNE import TSNE
from torchstat import stat
#DDGCN
###################

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type = int, required = True)
    args = parse.parse_args()
    config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
    config = Config(config_file)

    cuda = not config.no_cuda and torch.cuda.is_available()
    epoch_list=[]
    loss_list=[]
    acc_test_list = []
    acc_train_list = []
    atten_list =[]

    use_seed = config.no_seed
    if not use_seed:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if cuda:
            torch.cuda.manual_seed(config.seed)

   
    sadj, fadj = load_graph(args.labelrate, config)
    features, labels, idx_train, idx_test = load_data(config)
    # embedding = TSNE().fit(idx_train)
    
    model = SFGCN(nfeat = config.fdim,
              nhid1 = config.nhid1,
              nhid2 = config.nhid2,
              nclass = config.class_num,
              n = config.n,
              dropout = config.dropout)
    # stat(model,(22,sadj,fadj))
    if cuda:
        model.cuda()
        features = features.cuda()
        sadj = sadj.cuda()
        fadj = fadj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
 
    def train(model, epochs):
        model.train()
        optimizer.zero_grad()
        output, att, emb1, com1, com2, emb2, emb= model(features, sadj, fadj)
        # stat(model,(22,128,256))
        loss_class =  F.nll_loss(output[idx_train], labels[idx_train])
        loss_dep = (loss_dependence(emb1, com1, config.n) + loss_dependence(emb2, com2, config.n))/2
        loss_com = common_loss(com1,com2)
        loss = loss_class + config.beta * loss_dep + config.theta * loss_com
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        acc_test, macro_f1, emb_test, atten, output = main_test(model)
        print('epoch:{}'.format(epochs),
              'loss_train: {:.4f}'.format(loss.item()),
              'acc_train: {:.4f}'.format(acc.item()),
              'acc_test: {:.4f}'.format(acc_test.item()),
            #   'embedding:{:.4f}'.format(emb_test),
              'f1_test:{:.4f}'.format(macro_f1.item()))
        print('====================')
        print(len(atten.tolist()))
        # print(atten.tolist())
        atten_num=[]
        for i in atten.tolist():
            for j in i:
                # print(j)
                for x in j:
                    # print(x)
                    atten_num.append(x)

        feat_atten= [(x/26)+0.006 for x in atten_num]
        feat_atten=sorted(feat_atten) 
        print("Feature attnetion value:",feat_atten)
        epoch_list.append(epoch)
        loss_list.append(loss.item())
        acc_test_list.append(acc_test.item())
        acc_train_list.append(acc.item())
        label_dict = {0:"0",1:"1",2:"2",3:"3"} # 定义标签颜色字典
        # atten_list.append(atten.item())
        # atten = torch.tensor([item.detach().numpy() for item in atten])
        return loss.item(), acc_test.item(), macro_f1.item(), emb_test,epoch_list,loss_list,acc_test_list,acc_train_list, atten_list,output

    def main_test(model):
        model.eval()
        output, att, emb1, com1, com2, emb2, emb = model(features, sadj, fadj)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        label_max = []
        for idx in idx_test:
            label_max.append(torch.argmax(output[idx]).item())
        labelcpu = labels[idx_test].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1, emb, att, output
    
    # t-SNE 降维
    def t_SNE(output, dimention):
    # output:待降维的数据
    # dimention：降低到的维度
        tsne = manifold.TSNE(n_components=dimention, init='pca', random_state=0)
        result = tsne.fit_transform(output)
        return result

    # Visualization with visdom
    def Visualization(vis, result, labels,title):
        # vis: Visdom对象
        # result: 待显示的数据，这里为t_SNE()函数的输出
        # label: 待显示数据的标签
        # title: 标题
        vis.scatter(
            X = result,
            Y = labels+1,           # 将label的最小值从0变为1，显示时label不可为0
        opts=dict(markersize=4,title=title),
        )


    acc_max = 0
    f1_max = 0
    epoch_max = 0
    for epoch in range(config.epochs):
        loss, acc_test, macro_f1, emb,e_list,l_list,test_list,train_list,atten_value,out1 = train(model, epoch)
        if acc_test >= acc_max:
            acc_max = acc_test
            f1_max = macro_f1
            epoch_max = epoch
    print('epoch:{}'.format(epoch_max),
          'acc_max: {:.4f}'.format(acc_max),
          'f1_max: {:.4f}'.format(f1_max),
          'embedding'.format(emb))
    print(out1)
    print (e_list)
    print (l_list)
    print(atten_value)
    #计算预测值
    preds= out1.max(1)[1].type_as(labels)
    #output格式转换
    out1=out1.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()
    # Visualization with visdom
    # vis = Visdom(env='DDGCN Visualization')
    # # ground truth 与 预测结果 可视化
    # result_all_2d = t_SNE(out1, 2)
    # Visualization(vis, result_all_2d, labels,
    #               title='[ground truth of all samples]\n Dimension reduction to %dD' % (result_all_2d.shape[1]))
    # result_test_2d = t_SNE(out1[idx_test.cpu().detach().numpy()], 2)
    # Visualization(vis, result_test_2d, preds[idx_test.cpu().detach().numpy()],
    #               title='[prediction of test set]\n Dimension reduction to %dD' % (result_test_2d.shape[1]))

    # result_all_3d = t_SNE(out1, 3)
    # Visualization(vis, result_all_3d, labels,
    #               title='[ground truth of all samples]\n Dimension reduction to %dD' % (result_all_3d.shape[1]))
    # result_test_3d = t_SNE(out1[idx_test.cpu().detach().numpy()], 3)
    # Visualization(vis, result_test_3d, preds[idx_test.cpu().detach().numpy()],
    #               title='[prediction of test set]\n Dimension reduction to %dD' % (result_test_3d.shape[1]))
 
    # plt.subplot(2, 1, 2)
    # plt.plot(e_list, l_list, '.-',label="Loss")
    # # plt.plot(e_list, train_list, '.-',label="Train_Accuracy")
    # # plt.plot(e_list, test_list, '.-',label="Test_Accuracy")
    # plt.xlabel('epoch')
    # plt.ylabel('Loss')
    # plt.legend(loc='best')
    # plt.show() 



    
    
