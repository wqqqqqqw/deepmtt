#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:34:11 2018

@author: ljx
"""

#==============================================================================
#train the network with bidirectional lstm, train the detas of trajectory
#==============================================================================
# import numpy as np
# import tensorflow as tf
# from tensorflow.python.ops import variable_scope as vs

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import os
from pathlib import Path
from math import sqrt
from .training_data import *
# from maxout import max_out as MO     #maxout activation function
import argparse
from copy import copy
import random
from scipy.io import loadmat, savemat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "MTT_TrainingData.mat"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "pytorch" / "Save_Model"

def Noisy_af(Xt):                    #noisy activation function
    p = 1
    c = 1
    h = tf.nn.relu(0.5*Xt+0.5)-tf.nn.relu(0.5*Xt-0.5) -0.5
#    h = tf.nn.relu(Xt+1)-tf.nn.relu(Xt-1)-1
    y = h + c*tf.square(tf.sigmoid(p*(h-Xt))-0.5) * tf.random_normal(tf.shape(Xt),mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
    return y

def piecewise(Xt):                  #piecewise activation function
#    y = (tf.nn.relu(0.5*Xt+0.5)-tf.nn.relu(0.5*Xt-0.5)-0.5) #piecewise activation function
    y = tf.nn.relu(Xt+1)-tf.nn.relu(Xt-1)-1
    return y

def fir_filter(x,w,b_size,t_size): #快速FIR滤波
    with tf.name_scope('FIR_filter'):
        #x,待滤波的数据,shape为[batch_size,time_size,output_size]
        #w,滤波网络,shape为[fir_size, output_size]
#        shape_x = x.get_shape().as_list()
        shape_w = w.get_shape().as_list()
        x_add = tf.constant(0, dtype=tf.float32, shape=[b_size,shape_w[0]-1,shape_w[1]], name='X_add')
        x = tf.concat([x_add,x],1)  #给前面不足长度的待滤波序列补全零
        x = tf.expand_dims(x,2)     #扩展一维，存储待滤波数据
        y = []
        for i in range(shape_w[0]):
            y.append(x[:,i:i+t_size,:,:])
        z = tf.concat(y,2)
        z = z*w
        return tf.reduce_sum(z,reduction_indices=2)
#==============================================================================
#==============================================================================
#建立lstm网络

# from tensorflow.contrib import rnn
# Set CPU/GPF mode
#sess = tf.Session()  #CPU
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction=0.9
# sess = tf.Session(config=config)

#超参数的定义-----------------------------------------
# Set RNN parameter
BN = 2.0
lr = 1e-5
#lr = 1e3
#处理数据的batch大小
_batch_size = np.array([int(100*BN)])
# The size of batch for learning，这是在图里的batch_size张量定义

# 每个时刻的输入特征是4维的，就是每个时刻输入一行，一行有距离x,y和速度vx,vy
input_size = 4
# 时序持续长度
timestep_size = 50

# ## 隐含层的数量
# #hidden_size = 64
# # LSTM layer 的层数
# layer_num = 3
# #第一层隐层的节点数
# hidden_size_1 = 128
# #第二层隐层的节点数
# hidden_size_2 = 256
# #第三层隐层的节点数
# hidden_size_3 = 256
# #输出层maxout节点数
# maxout_size = 64
# #正则项系数
# lambda1 = 0.003
# #FIR滤波层滤波阶数
# fir_size = 5
#
# # 最后输出向量的维度
# output_size = 4



#************** Transformer Model
class FullAttention(nn.Module):
    '''
    The Attention operation
    '''

    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = out.view(B, L, -1)

        return self.out_projection(out)

class Transformer_block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super(Transformer_block, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        # self.pre_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        x = inputs

        # Multi-head attention
        time_enc = self.attention(x,x,x)

        # Dropout and residual
        time_enc = x + self.dropout(time_enc)

        # Layer Normalization
        y = self.norm1(time_enc)
        y = y + self.dropout(self.MLP1(time_enc))
        out = self.norm2(y)
        return out


class EntangleModel(torch.nn.Module):
    def __init__(self, d_model):
        super(EntangleModel, self).__init__()
        D = d_model
        self.FC_xs = torch.nn.Linear(D, D)
        self.FC_xt = torch.nn.Linear(D, D)
        self.FC_h1 = torch.nn.Linear(D, D)
        self.FC_h2 = torch.nn.Linear(D, D)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = self.sigmoid(torch.add(XS, XT))
        H = torch.add((z * HS), ((1 - z) * HT))
        H = self.FC_h2(F.relu(self.FC_h1(H)))
        return H


class Spatial_Temportal_Transformer(nn.Module):
    def __init__(self,d_model, n_heads, in_dim,seq_len,d_ff=None, dropout=0.1):
        super(Spatial_Temportal_Transformer, self).__init__()
        # self.enc_embedding_layer = nn.Linear(in_dim, d_model)
        # self.enc_pos = nn.Parameter(
        #     torch.randn(1, seq_len, d_model))
        # self.pre_norm = nn.LayerNorm(in_dim)
        # self.Spatial_Transformer = Transformer_block(d_model, n_heads,d_ff,dropout)
        self.Spatial_Transformer = Transformer_block(seq_len, n_heads, d_ff, dropout)
        self.Temporal_Transformer = Transformer_block(d_model, n_heads,d_ff,dropout)
        self.entangle = EntangleModel(d_model)
        # print('d_model',d_model)

    def forward(self, inputs,X_pos):
        # inputs = self.pre_norm(inputs)
        # X_embedding = self.enc_embedding_layer(inputs)
        # X_pos = inputs + self.enc_pos
        # print('X_pos',X_pos.shape)
        X_Temporal = self.Temporal_Transformer(X_pos)
        X_Spatial = self.Spatial_Transformer(inputs.transpose(1,2)) ### 维度
        X_ST = self.entangle(X_Temporal, X_Spatial.transpose(1,2))
        return X_ST
class MLP(nn.Module):

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim,
            num_layers,
            use_instancenorm=True,
            dropout_rate=0
    ):
        super().__init__()
        model = [nn.Linear(input_dim, hidden_dim)]
        if use_instancenorm:
            model += [nn.InstanceNorm1d(hidden_dim)]
        model += [nn.ReLU()]

        for _ in range(num_layers - 2):
            model += [nn.Linear(hidden_dim, hidden_dim)]
            if use_instancenorm:
                model += [nn.InstanceNorm1d(hidden_dim)]
            model += [nn.ReLU()]
        model += [nn.Linear(hidden_dim, output_dim)]

        self.model = nn.Sequential(*model)

    def forward(self, inps):
        return self.model(inps)

class Predict_Layer(nn.Module):
    def __init__(self, in_channels, seq_length, pred_length, n_features):
        super(Predict_Layer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3,padding=1)
        self.fc = MLP(input_dim=32 * (seq_length // 2), output_dim=pred_length * n_features, hidden_dim=128, num_layers=5, use_instancenorm=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Network(nn.Module):
    def __init__(self, encode_layers,d_model,n_heads,in_dim,seq_len,pred_length,d_ff=None,dropout=0.1):
        super(Network, self).__init__()
        self.ST_Transformer = torch.nn.ModuleList([Spatial_Temportal_Transformer(d_model,n_heads,in_dim,seq_len,d_ff=d_ff,dropout=dropout) for _ in range(encode_layers)])
        self.Predict_layer = Predict_Layer(in_channels=d_model,seq_length=seq_len,pred_length=pred_length,n_features=1)
        self.enc_embedding_layer = nn.Linear(in_dim, d_model)
        self.pre_norm = nn.LayerNorm(in_dim)
        self.enc_pos = nn.Parameter(
            torch.randn(1, seq_len, d_model))
    def forward(self,inputs):
        inputs = self.pre_norm(inputs)
        inputs = self.enc_embedding_layer(inputs)
        X_pos = inputs + self.enc_pos
        inputs = self.ST_Transformer[0](inputs, X_pos)
        for l in range(1, len(self.ST_Transformer)):
            inputs = self.ST_Transformer[l](inputs,inputs)
        inputs = inputs.transpose(1,2)
        X_pre = self.Predict_layer(inputs)
        return X_pre

##直接加载部分模型参数
##-----1,save the lstm network parameters
#saver_lstm = tf.train.Saver(lstm_variables)  
#model_path = "/home/ljx/proj/Maneuvering_Target_Tracking/model_save/LMTT1_deta/LMTT_vb.ckpt"
#saver_lstm.restore(sess, model_path)
##-----2,save the output layer
#saver_outputs = tf.train.Saver(lstm_outputs)
#model_path = "/home/ljx/proj/Maneuvering_Target_Tracking/model_save/LMTT1_deta/LMTT_op.ckpt"
#saver_outputs.restore(sess, model_path)

#迭代总次数
iter_time = 100000
# iter_time = 50
#显示准确率的相隔次数
accu_st = 10
save_st = 1000

#准确率存储，初始化
accuracy_save = np.array([0.0 for ttt in range(int(iter_time/accu_st))], 'float64')
itertime_save = np.array([0.0 for ttt in range(int(iter_time/accu_st))], 'float64')
t_0 = 0

#数据预处理1----整个batch数据中的最大值作为归一化  
def Data_Pro1(data):
    weight = np.max(np.abs(data))
    results = data/weight
    return results, weight

#数据处理2-----batch里面每一个数据按照第一个值的最大值进行归一化
# def Data_Pro2(data):
#     weight = np.max(np.abs(data[:,0,:]), axis=1)
#     weight = np.transpose(np.array([[weight]]),[2,1,0])
#     results = data/weight
#     return results, weight

def Data_Pro2(data):
    # weight = torch.max(torch.abs(data[:, 0, :]), dim=1)  # 形状 [batch_size]
    # weight = weight.view(-1, 1, 1)  # 转换为 [batch_size, 1, 1]
    weight = torch.max(torch.abs(data[:, 0, :]), dim=1).values  # 形状 [batch_size]
    weight = weight.clamp_min(1e-12).view(-1, 1, 1)  # 转换为 [batch_size, 1, 1]
    results = data / weight
    return results, weight

def data_iter(batch_size, ori_traj_set, obser_set, tra_traj_set, output_results_set, Traj_turn_set):
    target_samples = ori_traj_set.shape[0]  # 样本轨迹数
    Kindx = list(range(target_samples))
    random.shuffle(Kindx)
    for i_batch in range(0, target_samples, batch_size):
        j_batch = torch.LongTensor(Kindx[i_batch:min(i_batch + batch_size, target_samples)]).to(device)
        yield ori_traj_set.index_select(0, j_batch),obser_set.index_select(0, j_batch),tra_traj_set.index_select(0, j_batch), \
            output_results_set.index_select(0, j_batch), Traj_turn_set.index_select(0, j_batch)

def train(encode_layers,d_model,n_heads,in_dim,seq_len,pred_length,d_ff=None,dropout=0.1):
    print("model init")
    net = Network(encode_layers, d_model, n_heads, in_dim, seq_len,pred_length, d_ff, dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-4
    if torch.cuda.is_available():
        net.cuda()
    net.double()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                    lr=learning_rate)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for name, param in net.named_parameters():
        print("model:",name,param.requires_grad)

    ori_traj_set, obser_set, tra_traj_set, output_results_set, Traj_turn_set = creat_batch3(0, 100000 - 1, 30, 3, 300)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    savemat(DATA_PATH, {'ori_traj_set':ori_traj_set,'obser_set':obser_set,'tra_traj_set':tra_traj_set,'output_results_set':output_results_set,'Traj_turn_set':Traj_turn_set})
    ori_traj_set = torch.DoubleTensor(ori_traj_set).to(device)
    obser_set = torch.DoubleTensor(obser_set).to(device)
    tra_traj_set = torch.DoubleTensor(tra_traj_set).to(device)
    output_results_set = torch.DoubleTensor(output_results_set).to(device)
    Traj_turn_set = torch.DoubleTensor(Traj_turn_set).to(device)
    print('data OK')
    for i in range(iter_time):
        for ori_traj, obser, tra_traj, output_results, Traj_turn in data_iter(100,ori_traj_set, obser_set, tra_traj_set, output_results_set, Traj_turn_set):
            # ori_traj, obser, tra_traj, output_results, Traj_turn = creat_batch3(0, 100000 - 1, 30, 3, BN)
            my_batch, _ = Data_Pro2(ori_traj)
            detain = ori_traj[:, 1:, :] - ori_traj[:, :-1, :]
            X_pre = net(my_batch)
            loss = mse_loss(X_pre, Traj_turn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (i + 1) % accu_st == 0:
            print('loss',loss)
            X_pre = net(my_batch)
            rmse = torch.mean(torch.abs(Traj_turn - X_pre)).item()
            print("step %d, Tracking RMSE of Turn rate: %g" % (i + 1, rmse))

        if (i + 1) % save_st == 0:
            best_state_dict = copy(net.state_dict())
            Saved_dict = {'model': best_state_dict}
            torch.save(Saved_dict, CHECKPOINT_DIR / f"ST_Transformer_0418_iter_{i+1}.pth")

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode_layers",type=int,default=2)    # 1
    parser.add_argument("--d_model",type=int,default=32)
    parser.add_argument("--n_heads",type=int,default=3)
    parser.add_argument("--in_dim",type=int,default=4)
    parser.add_argument("--seq_len",type=int,default=50)
    parser.add_argument("--pred_length",type=int,default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train(encode_layers=args.encode_layers,d_model=args.d_model,n_heads=args.n_heads,in_dim=args.in_dim,\
          seq_len=args.seq_len,pred_length=args.pred_length)

if __name__ == "__main__":
    main()

#
# #==============================================================================
# #训练：========================================================================
# for i in range(iter_time):
# #    _,_,_, batch_input, output_results= creat_batch(_batch_size, timestep_size)
#     ori_traj,_,tra_traj,output_results = creat_batch3(0,100000-1,30,3,BN)
# #    ori_traj_c, ori_traj_f = data_change(ori_traj)
# #    tra_traj_c, _ = data_change(tra_traj)
# #    #数据处理1----整个batch数据中的最大值作为归一化
# #    my_batch, _ = Data_Pro1(tra_traj)
#
#     #数据处理2-----batch里面每一个数据按照第一个值的最大值进行衰减，线性衰减
#     my_batch, _ = Data_Pro2(tra_traj)
#     detain = ori_traj[:,1:,:]-ori_traj[:,:-1,:]
# #    output_results = ori_traj_c - tra_traj_c
#
#     if (i+1)%accu_st == 0:
#         my_results_t = sess.run(y_pre, feed_dict={
#             X:my_batch, y: output_results, keep_prob: 1.0, batch_size: _batch_size})
#         # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
#         print ("step %d, Tracking RMSE X: %g, Y: %g" % ((i+1), np.mean(np.abs(output_results-my_results_t)[:,:,0]), np.mean(np.abs(output_results-my_results_t)[:,:,1])))
#         accuracy_save[t_0] = np.mean(np.abs(output_results-my_results_t)[:,:,0:2])
#         itertime_save[t_0] = i+1
#         t_0 = t_0 +1
#
# #        my_tensor = sess.run(merged, feed_dict={
# #            X:my_batch, y: output_results, Xtrac:tra_traj, myd:detain, keep_prob: 1.0, batch_size: _batch_size})
# #        writer.add_summary(my_tensor, i)
#
#         myaccuracy = {'accuracy_save':accuracy_save,'itertime_save':itertime_save}
#         scio.savemat('my_accuracy_deta_bidr', myaccuracy)
#
#     if (i+1)%save_st == 0:
#         #继续保存模型
#         saver = tf.train.Saver()
#         model_path = "/home/ljx/文档/OpenSources/DeepMTT/Models/LMTT.ckpt"
# #        save_path = saver.save(sess, model_path, global_step=i)
#         save_path = saver.save(sess, model_path)
# #        print "Model saved in file: %s" % save_path
#
#
# #        #直接保存部分模型参数
# #        #-----1,save the lstm network parameters
# #        saver_lstm = tf.train.Saver(lstm_variables)
# #        model_path = "/home/ljx/proj/Maneuvering_Target_Tracking/model_save/LMTT1/LMTT_lstm.ckpt"
# #        saver_lstm.save(sess, model_path)
# #        #-----2,save the output layer
# #        saver_outputs = tf.train.Saver(lstm_outputs)
# #        model_path = "/home/ljx/proj/Maneuvering_Target_Tracking/model_save/LMTT1/LMTT_op.ckpt"
# #        saver_outputs.save(sess, model_path)
#
#     _ = sess.run(train_step, feed_dict={X:my_batch, y: output_results, keep_prob: 1.0, batch_size: _batch_size})
# #    _ = sess.run(train_deta, feed_dict={X:my_batch, y: output_results, Xtrac:tra_traj, myd:detain, keep_prob: 1.0, batch_size: _batch_size})

## 计算测试数据的准确率
#Traj_r,_, my_tracking_results,my_real_deta = creat_batch()
##my_tracking_results_c, my_tracking_results_f = data_change(my_tracking_results)
##Traj_r_c,_ = data_change(Traj_r)
##my_real_deta = Traj_r_c - my_tracking_results_c
##数据处理1/2-----batch里面每一个数据按照第一个值的最大值进行归一化
#my_tracking_inputs,_ = Data_Pro2(my_tracking_results)
#my_result = sess.run(y_pre, feed_dict={
#    X: my_tracking_inputs, y: my_real_deta, keep_prob: 1.0, batch_size: _batch_size})
##my_result_return = my_result + my_tracking_results_c
##my_traj_final = my_result_return * my_tracking_results_f
#print "test x orignal RMSE %g"% (np.mean(np.abs(my_real_deta)[:,:,0]))
#print "test x orignal RMSE %g"% (np.mean(np.abs(my_real_deta)[:,:,1]))
#print "test x RMSE %g"% (np.mean(np.abs(my_real_deta-my_result)[:,:,0]))
#print "test x RMSE %g"% (np.mean(np.abs(my_real_deta-my_result)[:,:,1]))
##画图
#import matplotlib.pyplot as plt
##plt.figure(1) # 创建图表1
##plt.plot(itertime_save, accuracy_save)
##plt.xlabel('Iteration Times')# make axis labels3
##plt.ylabel('Accuracy')
#my_traj_pre = my_tracking_results + my_result
#for i in range(10):
#    plt.figure(i) # 创建图表1
#    plt.plot(Traj_r[i,:,0], Traj_r[i,:,1], ".-")
#    plt.plot(my_traj_pre[i,:,0], my_traj_pre[i,:,1], "x-")
#    plt.plot(my_tracking_results[i,:,0], my_tracking_results[i,:,1], "^-")
##==============================================================================

##继续保存模型
#saver = tf.train.Saver()
#model_path = "/home/ljx/文档/Python_Proj/v1_0/model_save/TT2.ckpt"
#save_path = saver.save(sess, model_path)
#print "Model saved in file: %s" % save_path

##直接保存部分模型参数
##-----1,save the lstm network parameters
#saver_lstm = tf.train.Saver(lstm_variables)  
#model_path = "/home/ljx/文档/Python_Proj/v1_0/model_save/TT_ny_lstm.ckpt"
#saver_lstm.save(sess, model_path)
##-----2,save the output layer
#saver_outputs = tf.train.Saver(lstm_outputs)
#model_path = "/home/ljx/文档/Python_Proj/v1_0/model_save/TT_ny_outputs.ckpt"
#saver_outputs.save(sess, model_path)
