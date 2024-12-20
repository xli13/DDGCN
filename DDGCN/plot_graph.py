import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# topology_attention=[0.3956,0.4003,0.33125,0.3419,0.29275,0.49125,0.1949,0.2995,0.38935,0.39025,0.3921,0.3906,
#         0.3938,0.2983,0.28845,0.3892,0.4911,0.3896,0.49205,0.39645,0.3868,0.28765,0.2894]

# feature_attention= [0.6044,0.5997,0.66865,0.658,0.70725,0.50875,0.8051,0.7005,0.61065,0.60975,0.6079,0.6095
#         ,0.6061,0.7016,0.71155,0.6107,0.5089,0.6104,0.50785,0.60355,0.6132,0.71235,0.7106]
# data = pd.DataFrame({"Feature Doamin":feature_attention,"Topology Doamin":topology_attention})
# # print (len(topology_attention))
# data.boxplot()
# plt.ylabel("Attention")
# plt.xlabel("Domain")
# plt.title("Analysis of attention distributions")
# plt.show()
# epoch = []
# for i in range(60):
#     epoch.append(i)

# loss_list = [1.3953,1.3933,1.3921,1.3906,1.3887,1.3870,1.3850,1.3840,
# 1.3793,1.3770,1.3757,1.3744,1.3729,1.3713,1.3692,1.3674,
# 1.3661,1.3649,1.3627,1.3612,1.3605,1.3585,1.3571,1.3553,
# 1.3531,1.3517 ,1.3504 ,1.3478 ,1.3461 ,1.3454,1.3433,1.3422,
# 1.3400,1.3382 ,1.3370,1.3353,1.3336,1.3314,1.3290,1.3273,1.3253,1.3239,
# 1.3228,1.3205,1.3182,1.3172,1.3149,1.3127,1.3122,1.3102,1.3077,1.3058,1.3041,
# 1.2999,1.2993,1.2981,1.2958,1.2943,1.2929,1.2892]

# # print (len (loss_list))
# plt.plot(epoch,loss_list)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.show() 


# Combination loss
x =[1,2,3,4,5,6,7,8,9,10]
x_index =['0', '0.0001','0.001','0.01','0.1','1','10','100','1000','10000']
y1 = [89.83,90.22,92.28,94.39,91.76,90.83,87.50,86.11,85.83,84.17]
y2 = [92.14,95.52,96.36,97.13,95.28,96.11,93.46,90.28,90.44,87.54]
y3 = [96.61,97.20,97.78,98.02,98.61,97.17,98.43,95.33,94.39,91.44]
plt.plot(x,y1,label='15%',linewidth=1,color='r',marker='o',markersize=5)
plt.plot(x,y2,label='30%',linewidth=1,color='g',marker='x',markersize=5)
plt.plot(x,y3,label='60%',linewidth=1,color='b',marker='v',markersize=5)
plt.xlabel('Combination Constriant Coefficient',fontweight='bold')
plt.ylabel('Accuracy(%)',fontweight='bold')
plt.ylim((80,100))
plt.title('Analysis of Combination Constriant Coefficient')
plt.legend()
_ = plt.xticks(x,x_index)
plt.show()


# difference parameter
# x =[1,2,3,4,5,6,7,8,9,10,11,12]
# x_index =['0','1e-10','5e-10','1e-9','5e-9','1e-8','5e-8','1e-7','5e-7','1e-6','5e-6','1e-5']
# y1 =[88.15,89.87,90.37,92.80,93.10,93.83,86.17,84.60,82.83,79.67,77.00,73.83]
# y2 =[93.84,94.49,95.7,96.62,97.13,95.64,92.42,90.40,89.44,84.72,80.60,78.34]
# y3 =[92.80,94.60,97.78,98.04,98.65,97.28,94.32,92.84,90.06,88.94,87.83,83.42]
# plt.plot(x,y1,label='15%',linewidth=1,color='r',marker='o',markersize=5)
# plt.plot(x,y2,label='30%',linewidth=1,color='g',marker='x',markersize=5)
# plt.plot(x,y3,label='60%',linewidth=1,color='b',marker='v',markersize=5)
# plt.xlabel('Differnece Constriant Coefficient',fontweight='bold')
# plt.ylabel('Accuracy(%)',fontweight='bold')
# plt.ylim((65,100))
# plt.title('Analysis of Differnece Constriant Coefficient')
# plt.legend()
# _ = plt.xticks(x,x_index)
# plt.show()

# balance parameter
# x =[1,2,3,4,5,6,7,8,9]
# x_index =['0.1', '0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
# y1 = [88.42,89.68,89.93,90.76,91.03,91.53,92.40,94.39,90.23]
# y2 = [90.14,91.45,92.78,93.28,95.01,97.08,97.13,96.91,93.07]
# y3 = [93.24,93.92,94.80,95.46,96.28,97.78,98.20,98.67,95.33]
# plt.plot(x,y1,label='15%',linewidth=1,color='r',marker='o',markersize=5)
# plt.plot(x,y2,label='30%',linewidth=1,color='g',marker='x',markersize=5)
# plt.plot(x,y3,label='60%',linewidth=1,color='b',marker='v',markersize=5)
# plt.xlabel('Balance Parameter',fontweight='bold')
# plt.ylabel('Accuracy(%)',fontweight='bold')
# plt.ylim((80,100))
# plt.title('Analysis of Balance Parameter')
# plt.legend()
# _ = plt.xticks(x,x_index)
# plt.show()

# bar chart for variant comparision
# label = ["15%",'30%','60%']
# dd_gcn_T = ['91.14','94.28','95.17']
# dd_gcn_F = ['92.57','96.63','96.29']
# dd_gcn_diff = ['92.8','96.12','97.22']
# dd_gcn_comb = ['93.34','96.63','97.61']
# dd_gcn = ['93.6','96.8','98.26']

# x = np.arange(len(label))
# width = 0.1
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width*2, dd_gcn_T, width, label='DD-GCN-w/o(T)')
# rects2 = ax.bar(x - width+0.01, dd_gcn_F, width, label='DD-GCN-w/o(F)')
# rects3 = ax.bar(x + 0.02, dd_gcn_diff, width, label='DD-GCN-Diff')
# rects4 = ax.bar(x + width+ 0.03, dd_gcn_comb, width, label='DD-GCN-Comb')
# rects5 = ax.bar(x + width*2 + 0.04, dd_gcn, width, label='DD-GCN')

# ax.set_ylabel('Accuracy', fontsize=16)
# ax.set_xlabel('Label Rate', fontsize=16)
# ax.set_title('Analysis of variants')
# plt.yticks([t for t in range(0,13,2)],fontsize=12);
# ax.set_xticks(x)
# ax.set_xticklabels(label)
# ax.legend()

# fig.tight_layout()

# plt.show()

# line chart for attention
# x =[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
# # x_index =['0','1','2','4','135','136','137','114','210','211','329','485','564','681','880']
# x_index2 =['0','1','2','4','35','36','37','114','205','201','8','55','217','280','469','71','114','117','200','476']
# y1 = [0.001511,0.00132,0.001334,0.0016,0.001131,0.0009,0.001051,0.000973,0.000835,0.00121,0.00179,0.0013,0.00193,0.00206,0.003153]
# # y2 = [0.00197,0.0011,0.002134,0.0026,0.002531,0.0018,0.001258,0.001473,0.001235,0.00201,0.00161,0.001132,0.00176,0.00112,0.00198]
# plt.plot(x,y1,linewidth=1,color='r',marker='o',markersize=5)
# # plt.plot(x,y2,label='Topology Domain',linewidth=1,color='g',marker='v',markersize=5)
# # plt.plot(x,y3,label='60%',linewidth=1,color='b',marker='v',markersize=5)
# plt.vlines(22, 0.001, 0.004, colors = "b", linestyles = "dashed")
# plt.xlabel('Node ID',fontweight='bold')
# plt.ylabel('Attention',fontweight='bold')
# plt.ylim((0,0.0035))
# plt.title('Attention Value of Different Node')
# # plt.legend()
# plt.tick_params(axis='x', labelsize=8) 
# _ = plt.xticks(x,x_index2)
# plt.show()
#############################
