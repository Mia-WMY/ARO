import torch.nn as nn
class normalize(nn.Module):
    def __init__(self,x0,if_trainable=False):
        super().__init__()
        self.mean=nn.Parameter(x0.mean(),requires_grad=if_trainable)
        self.std=nn.Parameter(x0.std(),requires_grad=if_trainable)
        # print(x0.mean(),x0.std())
    def forward(self,x):
        return (x-self.mean)/self.std
    def inverse(self,x):
        return x*self.std+self.mean



class normalize_3d:
    def __init__(self,data,if_trainable=False):
        super().__init__()
        self.min_val = nn.Parameter(torch.min(data),requires_grad=if_trainable)
        self.max_val = nn.Parameter(torch.max(data),requires_grad=if_trainable)

    def forward(self, data):
        scaled_data = (data - self.min_val) / (self.max_val - self.min_val)
        return scaled_data

    def inverse(self, scaled_data):
        original_data = scaled_data * (self.max_val - self.min_val) + self.min_val
        return original_data
import torch
def cal_rmse(y_true,y_pred):
    mse=torch.mean((y_true-y_pred)**2)
    rmse=torch.sqrt(mse)
    return rmse.item()
################################################################
from torch.utils.data import Dataset, DataLoader
class CombinedDataset(Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.length = len(x)  # 假设 x, y, z 具有相同的长度

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]
