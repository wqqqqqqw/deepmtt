#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:34:11 2018

@author: ljx
"""

#==============================================================================
#train the network with bidirectional lstm, train the detas of trajectory
#==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from math import sqrt
from math import pi
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
DEFAULT_CHECKPOINT_PREFIX = "ST_Transformer_response_0705"
DT = 0.1

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


class KinematicTurnRateNet(nn.Module):
    def __init__(self, dt=DT):
        super().__init__()
        self.dt = dt
        self.scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, inputs):
        prev_v = inputs[:, -2, 2:4]
        curr_v = inputs[:, -1, 2:4]
        cross = prev_v[:, 0] * curr_v[:, 1] - prev_v[:, 1] * curr_v[:, 0]
        dot = prev_v[:, 0] * curr_v[:, 0] + prev_v[:, 1] * curr_v[:, 1]
        delta_heading = torch.atan2(cross, dot)
        turn_rate = delta_heading / self.dt * (180.0 / torch.pi)
        turn_rate = torch.clamp(turn_rate, -10.0, 10.0).unsqueeze(-1)
        return self.scale * turn_rate + self.bias


class FeatureTransformerNet(nn.Module):
    def __init__(self, encode_layers, d_model, n_heads, seq_len, pred_length, d_ff=None, dropout=0.1, dt=DT):
        super().__init__()
        self.dt = dt
        self.transformer = Network(
            encode_layers=encode_layers,
            d_model=d_model,
            n_heads=n_heads,
            in_dim=5,
            seq_len=seq_len,
            pred_length=pred_length,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(self, inputs):
        velocities = inputs[:, :, 2:4]
        prev_v = velocities[:, :-1, :]
        curr_v = velocities[:, 1:, :]
        cross = prev_v[:, :, 0] * curr_v[:, :, 1] - prev_v[:, :, 1] * curr_v[:, :, 0]
        dot = prev_v[:, :, 0] * curr_v[:, :, 0] + prev_v[:, :, 1] * curr_v[:, :, 1]
        delta_heading = torch.atan2(cross, dot)
        turn_feature = delta_heading / self.dt * (180.0 / torch.pi)
        turn_feature = torch.clamp(turn_feature, -10.0, 10.0)
        turn_feature = F.pad(turn_feature, (1, 0), value=0.0).unsqueeze(-1)
        return self.transformer(torch.cat([inputs, turn_feature], dim=-1))


def build_model(model_arch, encode_layers, d_model, n_heads, in_dim, seq_len, pred_length, d_ff=None, dropout=0.1):
    if model_arch == "transformer":
        return Network(encode_layers, d_model, n_heads, in_dim, seq_len, pred_length, d_ff, dropout)
    if model_arch == "feature_transformer":
        return FeatureTransformerNet(encode_layers, d_model, n_heads, seq_len, pred_length, d_ff, dropout)
    if model_arch == "kinematic":
        return KinematicTurnRateNet(DT)
    raise ValueError(f"Unsupported model architecture: {model_arch}")

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


def random_window(data, seq_len):
    full_len = data.shape[1]
    if seq_len > full_len:
        raise ValueError(f"seq_len ({seq_len}) must not exceed data length ({full_len})")
    if seq_len == full_len:
        return data
    start = random.randint(0, full_len - seq_len)
    return data[:, start:start + seq_len, :]


def data_iter(batch_size, ori_traj_set, obser_set, tra_traj_set, output_results_set, Traj_turn_set):
    target_samples = ori_traj_set.shape[0]  # 样本轨迹数
    Kindx = list(range(target_samples))
    random.shuffle(Kindx)
    for i_batch in range(0, target_samples, batch_size):
        j_batch = torch.LongTensor(Kindx[i_batch:min(i_batch + batch_size, target_samples)]).to(device)
        yield ori_traj_set.index_select(0, j_batch),obser_set.index_select(0, j_batch),tra_traj_set.index_select(0, j_batch), \
            output_results_set.index_select(0, j_batch), Traj_turn_set.index_select(0, j_batch)


def response_data_iter(batch_size, traj_set, turn_set):
    target_samples = traj_set.shape[0]
    indices = torch.randperm(target_samples, device=traj_set.device)
    for i_batch in range(0, target_samples, batch_size):
        batch_idx = indices[i_batch:min(i_batch + batch_size, target_samples)]
        yield traj_set.index_select(0, batch_idx), turn_set.index_select(0, batch_idx)


def load_or_create_training_data(data_path, use_existing_data=True):
    required_keys = (
        "ori_traj_set",
        "obser_set",
        "tra_traj_set",
        "output_results_set",
        "Traj_turn_set",
    )
    data_path = Path(data_path)

    if use_existing_data and data_path.exists():
        print(f"Loading existing training data: {data_path}")
        mat_data = loadmat(data_path)
        missing_keys = [key for key in required_keys if key not in mat_data]
        if missing_keys:
            raise KeyError(f"Training data file is missing keys: {missing_keys}")
        return tuple(mat_data[key] for key in required_keys)

    print(f"Generating new training data: {data_path}")
    training_data = creat_batch3(0, 100000 - 1, 30, 3, 300)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(data_path, dict(zip(required_keys, training_data)))
    return training_data


def state_transition_matrix(turn_rate_deg, dt=DT):
    omega = np.deg2rad(turn_rate_deg)
    if abs(omega) < 1e-12:
        return np.array(
            [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    s = np.sin(omega * dt)
    c = np.cos(omega * dt)
    return np.array(
        [[1.0, 0.0, s / omega, (c - 1.0) / omega],
         [0.0, 1.0, -(c - 1.0) / omega, s / omega],
         [0.0, 0.0, c, -s],
         [0.0, 0.0, s, c]],
        dtype=np.float32,
    )


def process_noise_matrix(state_noise_std=10.0, dt=DT):
    s_var = np.square(state_noise_std)
    t2 = dt * dt
    t3 = t2 * dt
    t4 = t2 * t2
    return np.array(
        [[t4 / 4.0, 0.0, t3 / 2.0, 0.0],
         [0.0, t4 / 4.0, 0.0, t3 / 2.0],
         [t3 / 2.0, 0.0, t2, 0.0],
         [0.0, t3 / 2.0, 0.0, t2]],
        dtype=np.float32,
    ) * s_var


def generate_response_batch(batch_size, seq_len, data_len=120, segment_count=4):
    windows = np.zeros((batch_size, seq_len, 4), dtype=np.float32)
    labels = np.zeros((batch_size, 1), dtype=np.float32)
    base_len, remainder = divmod(data_len, segment_count)
    lengths = [base_len + (1 if idx < remainder else 0) for idx in range(segment_count)]
    change_points = np.cumsum(lengths[:-1])

    for batch_idx in range(batch_size):
        rng = np.random.default_rng()
        start_distance = rng.uniform(0.5 * 1.852e3, 20.0 * 1.852e3)
        start_direction = rng.uniform(-pi, pi)
        start_speed = rng.uniform(-340.0, 340.0)
        velocity_direction = rng.uniform(-pi, pi)
        state = np.array(
            [
                start_distance * np.cos(start_direction),
                start_distance * np.sin(start_direction),
                start_speed * np.cos(velocity_direction),
                start_speed * np.sin(velocity_direction),
            ],
            dtype=np.float32,
        )

        turn_rates = [0.0]
        for _ in range(segment_count - 1):
            turn = 0.0
            while abs(turn) < 1e-6:
                turn = rng.integers(-100, 101) / 10.0
            turn_rates.append(float(turn))

        noise_cov = process_noise_matrix(rng.uniform(8.0, 13.0), DT)
        chol = np.linalg.cholesky(noise_cov + 1e-6 * np.eye(4, dtype=np.float32)).astype(np.float32)
        traj = np.zeros((data_len, 4), dtype=np.float32)
        turn_seq = np.zeros((data_len,), dtype=np.float32)
        for step_idx in range(data_len):
            segment_idx = int(np.searchsorted(change_points, step_idx, side="right"))
            turn_rate = turn_rates[segment_idx]
            traj[step_idx] = state
            turn_seq[step_idx] = turn_rate
            state = state_transition_matrix(turn_rate, DT) @ state
            if segment_idx > 0:
                state = state + chol @ rng.normal(size=4).astype(np.float32)

        end_idx = int(rng.integers(seq_len - 1, data_len))
        windows[batch_idx] = traj[end_idx - seq_len + 1:end_idx + 1]
        labels[batch_idx, 0] = turn_seq[end_idx]

    return windows, labels


def train(
    encode_layers,
    d_model,
    n_heads,
    in_dim,
    seq_len,
    pred_length,
    d_ff=None,
    dropout=0.1,
    use_existing_data=True,
    data_path=DATA_PATH,
    iterations=iter_time,
    batch_size=100,
    learning_rate=1e-4,
    checkpoint_prefix=DEFAULT_CHECKPOINT_PREFIX,
    checkpoint_dir=CHECKPOINT_DIR,
    save_every=save_st,
    dtype="float32",
    training_mode="response",
    model_arch="transformer",
    steps_per_epoch=50,
    response_samples=20000,
    response_data_len=120,
    response_segments=4,
):
    print("model init")
    net = build_model(model_arch, encode_layers, d_model, n_heads, in_dim, seq_len,pred_length, d_ff, dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_dtype = torch.float32 if dtype == "float32" else torch.float64
    net = net.to(device=device, dtype=tensor_dtype)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                    lr=learning_rate)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name, param in net.named_parameters():
        print("model:",name,param.requires_grad)

    if training_mode == "existing":
        ori_traj_set, obser_set, tra_traj_set, output_results_set, Traj_turn_set = load_or_create_training_data(
            data_path,
            use_existing_data=use_existing_data,
        )
        ori_traj_set = torch.as_tensor(ori_traj_set, dtype=tensor_dtype, device=device)
        obser_set = torch.as_tensor(obser_set, dtype=tensor_dtype, device=device)
        tra_traj_set = torch.as_tensor(tra_traj_set, dtype=tensor_dtype, device=device)
        output_results_set = torch.as_tensor(output_results_set, dtype=tensor_dtype, device=device)
        Traj_turn_set = torch.as_tensor(Traj_turn_set, dtype=tensor_dtype, device=device)
        print('data OK')
    else:
        print(f'generating response training windows: {response_samples}')
        response_traj, response_turn = generate_response_batch(
            response_samples,
            seq_len,
            data_len=response_data_len,
            segment_count=response_segments,
        )
        response_traj_set = torch.as_tensor(response_traj, dtype=tensor_dtype, device=device)
        response_turn_set = torch.as_tensor(response_turn, dtype=tensor_dtype, device=device)
        print('response data OK')
    model_config = {
        "encode_layers": encode_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "in_dim": in_dim,
        "seq_len": seq_len,
        "pred_length": pred_length,
        "d_ff": d_ff,
        "dropout": dropout,
        "dtype": dtype,
        "training_mode": training_mode,
        "model_arch": model_arch,
    }
    for i in range(iterations):
        if training_mode == "existing":
            batches = data_iter(batch_size,ori_traj_set, obser_set, tra_traj_set, output_results_set, Traj_turn_set)
        else:
            batches = response_data_iter(batch_size, response_traj_set, response_turn_set)

        for batch in batches:
            if training_mode == "existing":
                ori_traj, obser, tra_traj, output_results, Traj_turn = batch
                ori_traj = random_window(ori_traj, seq_len)
            else:
                ori_traj, Traj_turn = batch

            my_batch, _ = Data_Pro2(ori_traj)
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

        if save_every > 0 and (i + 1) % save_every == 0:
            best_state_dict = copy(net.state_dict())
            Saved_dict = {'model': best_state_dict, 'model_config': model_config}
            torch.save(Saved_dict, checkpoint_dir / f"{checkpoint_prefix}_iter_{i+1}.pth")

    best_state_dict = copy(net.state_dict())
    Saved_dict = {'model': best_state_dict, 'model_config': model_config}
    final_path = checkpoint_dir / f"{checkpoint_prefix}_iter_{iterations}.pth"
    torch.save(Saved_dict, final_path)
    print(f"Saved final checkpoint: {final_path}")

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-arch", choices=("transformer", "feature_transformer", "kinematic"), default="transformer")
    parser.add_argument("--encode_layers",type=int,default=2)    # 1
    parser.add_argument("--d_model",type=int,default=32)
    parser.add_argument("--n_heads",type=int,default=3)
    parser.add_argument("--in_dim",type=int,default=4)
    parser.add_argument("--seq_len",type=int,default=20)
    parser.add_argument("--pred_length",type=int,default=1)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint-prefix", type=str, default=DEFAULT_CHECKPOINT_PREFIX)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--training-mode", choices=("existing", "response"), default="response")
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--response-samples", type=int, default=20000)
    parser.add_argument("--response-data-len", type=int, default=120)
    parser.add_argument("--response-segments", type=int, default=4)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH, help="Path to training data .mat file")
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--use-existing-data",
        dest="use_existing_data",
        action="store_true",
        help="Load --data-path when it exists, otherwise generate it",
    )
    data_group.add_argument(
        "--regenerate-data",
        dest="use_existing_data",
        action="store_false",
        help="Ignore existing --data-path and regenerate training data",
    )
    parser.set_defaults(use_existing_data=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train(encode_layers=args.encode_layers,d_model=args.d_model,n_heads=args.n_heads,in_dim=args.in_dim,\
          seq_len=args.seq_len,pred_length=args.pred_length,d_ff=args.d_ff,dropout=args.dropout,\
          use_existing_data=args.use_existing_data,data_path=args.data_path,iterations=args.iterations,\
          batch_size=args.batch_size,learning_rate=args.learning_rate,\
          checkpoint_prefix=args.checkpoint_prefix,checkpoint_dir=args.checkpoint_dir,\
          save_every=args.save_every,dtype=args.dtype,training_mode=args.training_mode,\
          model_arch=args.model_arch,steps_per_epoch=args.steps_per_epoch,response_samples=args.response_samples,\
          response_data_len=args.response_data_len,\
          response_segments=args.response_segments)

if __name__ == "__main__":
    main()
