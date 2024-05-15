"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which takes the 2D spatial + 1D temporal equation directly as a 3D problem
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer


import os
from timeit import default_timer
from fourier_mcdropout.utilities3 import *
from fourier_mcdropout.Adam import Adam
from fourier_mcdropout.normalize import normalize,cal_rmse
from sklearn.metrics import mean_squared_error,r2_score
################################################################
# 3d z_fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width,dr=0.1):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        # self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width)
        # input channel is 4: the solution of the first 1 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)
        self.drop1 = nn.Dropout(dr)
        self.drop2 = nn.Dropout(dr)
        self.drop3 = nn.Dropout(dr)
        self.drop4 = nn.Dropout(dr)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = x.to(torch.float32)
        # print(x.shape) # 20 52 52 9 4
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        # print(x1.is_complex(), x2.is_complex)
        x = F.gelu(x)
        x=self.drop1(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.drop2(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x = self.drop3(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop4(x)
        x = self.fc2(x)
        return x


    def train_model(self,x_train,y_train,epochs,batch_size,work_dir):
            step_size = 100
            gamma = 0.5
            learning_rate = 0.001

            optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            x_normalizer = normalize(x_train)
            x_train = x_normalizer.forward(x_train)


            y_normalizer = normalize(y_train)
            y_train = y_normalizer.forward(y_train)

            train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                                       batch_size=batch_size,
                                                       shuffle=True, drop_last=False)

            myloss = LpLoss(size_average=False)

            y_normalizer.cuda()
            t1 = default_timer()
            for ep in range(epochs):

                train_l2 = 0
                for x, y in train_loader:
                    x, y = x.cuda(), y.cuda()

                    optimizer.zero_grad()
                    # out = model(x).reshape(batch_size, s, s)
                    x = x.to(dtype=torch.double)
                    out = self(x)
                    out=out.reshape(x.shape[0], x.shape[1], x.shape[2],x.shape[3])
                    out = y_normalizer.inverse(out)
                    y = y_normalizer.inverse(y)

                    loss = myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
                    loss.backward()

                    optimizer.step()
                    train_l2 += loss.item()

                scheduler.step()

            t2 = default_timer()
            print("training times:",t2-t1)
            current_path=os.getcwd()
            folder_path=os.path.join(current_path,work_dir)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            return x_normalizer,y_normalizer,folder_path



    def test_model(self,x_test,y_test,x_normalizer,y_normalizer):
            x_test = x_normalizer.forward(x_test)
            batch_size = 20
            test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                                      shuffle=False, drop_last=False)
            rmse_total = 0
            r2_total = 0
            maxx_total = 0
            differ_total = 0
            mape_total = 0
            pape_total = 0
            iter=0
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                mean_out,std_out=self.predict(x,y_normalizer)
                mean_out=torch.cat(mean_out,dim=0)
                out =mean_out.cpu().detach()
                y = y.cpu().detach().numpy()
                for i in range(x.shape[0]):
                    rmse=0
                    maxx=0
                    differ=0
                    mape=0
                    pape=0
                    for l in range(y.shape[-1]):
                        rmse += cal_rmse(out[i, ...,l], y[i, ...,l])
                        maxx += torch.max(torch.abs(out[i, ..., l] - y[i, ..., l]))
                        differ += torch.mean(torch.abs(out[i, ..., l] - y[i, ..., l]))
                        mape += torch.mean(torch.abs(out[i, ..., l] - y[i, ..., l]) / y[i, ..., l]) * 100
                        pape += torch.max(torch.abs(out[i, ..., l] - y[i, ..., l]) / y[i, ..., l]) * 100
                    rmse = rmse/y.shape[-1]
                    maxx = maxx / y.shape[-1]
                    differ = differ / y.shape[-1]
                    mape = mape / y.shape[-1]
                    pape = pape / y.shape[-1]
                    r2 = 0
                    for l in range(y.shape[-1]):
                        r2 += r2_score(out[i, ..., l].flatten(), y[i, ..., l].flatten())
                    r2 = r2 / y.shape[-1]
                    rmse_total += rmse
                    r2_total += r2
                    maxx_total += maxx
                    differ_total += differ
                    mape_total+=mape
                    pape_total+=pape
                    iter+=1
            rmse_total /= iter
            r2_total /= iter
            maxx_total /= iter
            differ_total /= iter
            pape_total /=iter
            mape_total /=iter
            print("rmse:", rmse_total)
            print("r2:", r2_total)
            print("max.terr:", maxx_total)
            print("avg.terr:", differ_total)
            print("mape:", mape_total)
            print("pape:", pape_total)

    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def predict(self,x,y_normalizer,samples=20):
            pred_mean = []
            pred_std=[]
            self.enable_dropout()
            for i in range(x.shape[0]):
                test_x=x[i,...]
                # print("test_x",test_x.shape)# 85 85 3 4
                test_x=test_x.unsqueeze(0) # 1 85 85 3 4
                result=[]
                for t in range(samples):
                    out = self(test_x).reshape(test_x.shape[0],test_x.shape[1],test_x.shape[2],test_x.shape[3])
                    out = y_normalizer.inverse(out)
                    out = out.cpu().detach()
                    result.append(out)
                stacked_tensor=torch.stack(result,axis=0)
                mean=torch.mean(stacked_tensor,axis=0)
                std=torch.std(stacked_tensor,axis=0)
                pred_mean.append(mean)
                pred_std.append(std)
            return pred_mean,pred_std
    def update_model(self,x_train,y_train,x_normalizer,y_normalizer):
        step_size = 50
        gamma = 0.5
        epochs=100
        batch_size = 20
        learning_rate = 0.0005

        optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        x_train = x_normalizer.forward(x_train)
        y_train = y_normalizer.forward(y_train)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                                   batch_size=batch_size,
                                                   shuffle=True, drop_last=False)


        myloss = LpLoss(size_average=False)
        y_normalizer.cuda()

        for ep in range(epochs):
            epoch_loss=0
            # t1 = default_timer()
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                # out = model(x).reshape(batch_size, s, s)
                x = x.to(dtype=torch.double)
                out = self(x).reshape(x.shape[0], x.shape[1], x.shape[2],x.shape[3])
                out = y_normalizer.inverse(out)
                y = y_normalizer.inverse(y)

                loss = myloss(out.view(x.shape[0], -1), y.view(x.shape[0], -1))
                loss.backward()

                optimizer.step()

            scheduler.step()

