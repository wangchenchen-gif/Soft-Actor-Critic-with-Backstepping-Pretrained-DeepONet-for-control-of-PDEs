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
import joblib
from pathlib import Path
#print("DeepXDE版本:", dde.__version__)

# ------------------- 加载训练好的scaler -------------------

N1 = 101


def preprocess_obs(obs: torch.Tensor) -> torch.Tensor:
    current_dir = Path(__file__).parent.resolve()
    
    obs_np = obs.detach().cpu().numpy()
    channel1 = obs_np[:, :N1]
    channel2 = obs_np[:, N1:]
    
    obs_scaled_np = np.hstack((channel1, channel2))
    obs_scaled = torch.from_numpy(obs_scaled_np).to(obs.device)
    return obs_scaled


def convert_to_tensor(input_data):
    # 检查输入是否为 PyTorch Tensor
    if not isinstance(input_data, torch.Tensor):
        # 如果输入是 NumPy 数组，将其转换为 Tensor
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data)
        else:
            # 如果输入既不是 Tensor 也不是 NumPy 数组，尝试直接转换为 Tensor
            input_data = torch.tensor(input_data)
    return input_data


def myReshapeFunc2(all_data, N1=101):  # 56中每个通道22点，总输入44维
    batch = all_data.shape[0]  # 获取批量大小
    # 拆分输入为两个通道：前N1维（λ/参数通道）和后N1维（状态u通道）
    channel1 = all_data[:, :N1].unsqueeze(1)  # (batch, 1, N1)
    channel2 = all_data[:, N1:2*N1].unsqueeze(1)  # (batch, 1, N1)
    # 合并为2通道一维张量，适配Conv1d输入格式(N, C, L)
    x_reshaped = torch.cat([channel1, channel2], dim=1)  # (batch, 2, N1)
    return x_reshaped


spatial1 = np.linspace(0, 1, 101, dtype=np.float32)
grid_1d = spatial1.reshape(-1, 1)  # 形状：(201, 1)
grid2 = torch.from_numpy(grid_1d).cuda()



class BranchNet(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.conv1 = torch.nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=2)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.fc1 = torch.nn.Linear(64 * 23, 101)  # 输出维度101
        
    def forward(self, x):
        x = x.reshape(x.shape[0], 2, self.shape)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
       
        return x

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=101):
        super().__init__(observation_space, features_dim)
        global grid2
        N1 = 101
        current_dir = Path(__file__).parent.resolve()
        model_path = current_dir / "modedlNeth2.pt"
        
        # 创建模型
        self.net1 = dde.nn.DeepONetCartesianProd([N1, BranchNet(N1)], [1, 32,64, 101], "relu", "Glorot normal").cuda()
        
        try:
            state_dict = torch.load(str(model_path), map_location=torch.device('cuda'))
            
            # 提取Branch网络权重
            branch_state_dict = {
                k.replace("net1.branch.", ""): v 
                for k, v in state_dict.items() 
                if k.startswith("net1.branch.")
            }
            
            # 提取Trunk网络权重
            trunk_state_dict = {
                k.replace("net1.trunk.", ""): v 
                for k, v in state_dict.items() 
                if k.startswith("net1.trunk.")
            }
            
            # 分别加载权重
            if hasattr(self.net1, 'branch'):
                self.net1.branch.load_state_dict(branch_state_dict, strict=False)
                print(f"成功加载Branch网络权重: {len(branch_state_dict)}个参数")
            
            if hasattr(self.net1, 'trunk'):
                self.net1.trunk.load_state_dict(trunk_state_dict, strict=False)
                print(f"成功加载Trunk网络权重: {len(trunk_state_dict)}个参数")
                
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            print(f"尝试加载的路径: {model_path}")
    def forward(self, observations):
        
        x = self.net1((observations, grid2))
        
        
        x = x.view(x.size(0), -1)
       
        
        x = torch.relu(x)
     
        
      
        return x

class BranchFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=101):
        super().__init__(observation_space, features_dim)
        self.net1 = BranchNet(101).cuda()
    
    def forward(self, observations):
        x = self.net1(observations)
        x = x.view(x.size(0), -1)
        x = torch.relu(x)
       
        return x
