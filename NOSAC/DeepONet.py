import deepxde as dde
import torch
import os
import numpy as np
from torch import nn
import torch.nn.functional as F
from gymnasium import spaces
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import create_mlp
import torch.optim as optim
# 假设lam和u具有相同的网格尺寸
grid = np.linspace(0, 1, 102, dtype=np.float32).reshape(102, 1)
grid = torch.from_numpy(np.array(grid)).cuda()

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=102):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # 分别为lam和u创建DeepONet
        self.net_lam = dde.nn.DeepONetCartesianProd([102, 256], [1,128,256], "relu", "Glorot normal").cuda()
        self.net_u = dde.nn.DeepONetCartesianProd([102, 256], [1,128,256], "relu", "Glorot normal").cuda()
        
        # 合并特征后的处理层
        self.fc = nn.Sequential(
            nn.Linear(204,128),  # 假设每个DeepONet输出256个特征
            nn.ReLU(),
            nn.Linear(128,features_dim),  # 假设每个DeepONet输出256个特征
            nn.ReLU(),
            
            
            
            
        ).cuda()

    def forward(self, observations):
        # 分离lam和u
        lam = observations[:, :102]  # 假设前202个值是lam
        u = observations[:, 102:]    # 假设后202个值是u
        
        # 分别通过各自的DeepONet处理
        x_lam = self.net_lam((lam, grid))
        x_u = self.net_u((u, grid))
        
        # 展平并合并特征
        x_lam = x_lam.view(x_lam.size(0), -1)
        x_u = x_u.view(x_u.size(0), -1)
        
        # 合并特征并通过全连接层
        x = torch.cat([x_lam, x_u], dim=1)
        x = self.fc(x)
        
        return x             # (B, 102)                 # (B, 102)
