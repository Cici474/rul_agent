import torch
import torch.nn as nn
import numpy as np
import math
import lightgbm as lgb
from config import LIGHTGBM_PARAMS

class Expert_LightGBM:
    def __init__(self):
        self.model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
        
    def _build_features(self, X, Macro_X):
        # 提取时间窗口内的统计特征（均值、标准差、最大值、最小值）和最后一个时间步的值（当前最新状态）
        X_mean = X.mean(axis=1)
        X_std = X.std(axis=1)
        X_max = X.max(axis=1)
        X_min = X.min(axis=1)
        X_last = X[:, -1, :] 
        
        # 将统计特征与传入的宏观特征(趋势、方差、绝对偏差)拼接
        return np.concatenate([X_mean, X_std, X_max, X_min, X_last, Macro_X], axis=1)

    def fit(self, X, Macro_X, y):
        features = self._build_features(X, Macro_X)
        self.model.fit(features, y)
        
    def forward(self, X, Macro_X):
        features = self._build_features(X, Macro_X)
        return self.model.predict(features)

class ECA_Block(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x) 
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        return x * self.sigmoid(y).expand_as(x)

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1, bias=False)
        )
        self.scale = 1.0 / (hidden_dim ** 0.5)
    def forward(self, x):
        weights = torch.softmax(self.attention(x) * self.scale, dim=1)
        return torch.sum(x * weights, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.BatchNorm1d(n_outputs), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.BatchNorm1d(n_outputs), nn.GELU(), nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Expert_TCN(nn.Module):
    def __init__(self, in_channels, seq_len):
        super().__init__()
        self.sensor_attention = ECA_Block(in_channels)
        self.input_conv = nn.Sequential(nn.Conv1d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.GELU())
        num_channels = [32, 64, 64, 128] 
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch = 32 if i == 0 else num_channels[i-1]
            layers += [TemporalBlock(in_ch, num_channels[i], 3, stride=1, dilation=dilation_size, padding=2*dilation_size, dropout=0.35)]
        self.network = nn.Sequential(*layers)
        self.attention = SelfAttention(num_channels[-1])
        self.layer_norm = nn.LayerNorm(num_channels[-1]) 
        self.fc = nn.Sequential(nn.Linear(num_channels[-1], 32), nn.GELU(), nn.Dropout(0.35), nn.Linear(32, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(self.input_conv(self.sensor_attention(x))).transpose(1, 2)
        return self.fc(self.layer_norm(self.attention(out))).squeeze(-1)

class Expert_BiLSTM(nn.Module):
    def __init__(self, in_channels, seq_len):
        super().__init__()
        self.sensor_attention = ECA_Block(in_channels)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.35),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.35)
        )
        self.lstm = nn.LSTM(64, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.35)
        self.attention = SelfAttention(128)
        self.fc = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Dropout(0.35), nn.Linear(32, 1))

    def forward(self, x):
        x = self.conv(self.sensor_attention(x.transpose(1, 2))).transpose(1, 2)
        out, _ = self.lstm(x)            
        return self.fc(self.attention(out)).squeeze(-1)

class Expert_Transformer(nn.Module):
    def __init__(self, in_channels, seq_len):
        super().__init__()
        self.d_model = 64 
        
        self.spatial_attention = ECA_Block(in_channels)
        self.spatial_conv = nn.Sequential(nn.Conv1d(in_channels, self.d_model // 2, 1), nn.BatchNorm1d(self.d_model // 2), nn.GELU())
        
        out_c = self.d_model // 2
        c1, c2 = out_c // 3, out_c // 3
        c3 = out_c - c1 - c2
        self.temporal_conv1 = nn.Conv1d(in_channels, c1, 3, padding=1)
        self.temporal_conv2 = nn.Conv1d(in_channels, c2, 5, padding=2)
        self.temporal_conv3 = nn.Conv1d(in_channels, c3, 7, padding=3)
        self.temporal_bn = nn.BatchNorm1d(out_c)

        self.pos_encoder = PositionalEncoding(self.d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, dropout=0.35, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.attention = SelfAttention(self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.fc = nn.Sequential(nn.Linear(self.d_model, 32), nn.GELU(), nn.Dropout(0.35), nn.Linear(32, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        x_spatial = self.spatial_conv(self.spatial_attention(x))
        x_temporal = torch.nn.functional.gelu(self.temporal_bn(torch.cat([
            self.temporal_conv1(x), self.temporal_conv2(x), self.temporal_conv3(x)
        ], dim=1)))
        
        x_fused = torch.cat([x_spatial, x_temporal], dim=1).transpose(1, 2) 
        out = self.transformer(self.pos_encoder(x_fused))
        return self.fc(self.layer_norm(self.attention(out))).squeeze(-1)