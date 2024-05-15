"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
import os
from timeit import default_timer
from fourier_mcdropout.utilities3 import *
from fourier_mcdropout.Adam import Adam
from fourier_mcdropout.normalize import normalize,cal_rmse
from sklearn.metrics import mean_squared_error,r2_score
torch.manual_seed(0)
np.random.seed(0)


################################################################
# z_fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width, dr=0.1):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        # DROPOUT
        self.drop1 =nn.Dropout(dr)
        self.drop2 = nn.Dropout(dr)
        self.drop3 = nn.Dropout(dr)
        self.drop4 = nn.Dropout(dr)
        # self.data_processor=normalize()

        self.s1=85
        self.s2=85


    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x=x.to(torch.float32)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x=self.drop1(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x=self.drop2(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x=self.drop3(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x=self.drop4(x)
        x = self.fc2(x)
        return x

    def train_model(self,x_train,x_test,y_train,y_test,ntrain,ntest,work_dir):
        step_size = 100
        gamma = 0.5
        epochs=500
        batch_size = 20
        learning_rate = 0.001

        optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        x_normalizer = normalize(x_train)
        x_train = x_normalizer.forward(x_train)
        x_test = x_normalizer.forward(x_test)

        y_normalizer = normalize(y_train)
        y_train = y_normalizer.forward(y_train)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                                   batch_size=batch_size,
                                                   shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                                  shuffle=False, drop_last=True)

        myloss = LpLoss(size_average=False)
        train_loss = []
        test_loss = []
        epoch = []
        y_normalizer.cuda()
        for ep in range(epochs):
            t1 = default_timer()
            train_l2 = 0
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()

                optimizer.zero_grad()
                # out = model(x).reshape(batch_size, s, s)
                x = x.to(dtype=torch.double)
                out = self(x).reshape(batch_size, x.shape[1], x.shape[2])
                out = y_normalizer.inverse(out)
                y = y_normalizer.inverse(y)

                loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
                loss.backward()

                optimizer.step()
                train_l2 += loss.item()

            scheduler.step()
            test_l2 = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()
                    out = self(x).reshape(batch_size, x.shape[1], x.shape[2])
                    out = y_normalizer.inverse(out)
                    test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

            train_l2 /= ntrain
            test_l2 /= ntest

            t2 = default_timer()
            print(ep, t2 - t1, train_l2, test_l2)
            train_loss.append(train_l2)
            test_loss.append(test_l2)
            epoch.append(ep)
        plt.plot(epoch, train_loss, label='train loss')
        plt.plot(epoch, test_loss, label='valid loss')
        plt.legend()
        current_path=os.getcwd()
        folder_path=os.path.join(current_path,work_dir)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        image_path=os.path.join(folder_path,'loss.png')
        plt.savefig(image_path)
        return x_normalizer,y_normalizer,folder_path



    def test_model(self,x_test,y_test,x_normalizer,y_normalizer):
        x_test = x_normalizer.forward(x_test)
        batch_size = 100
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                                  shuffle=False, drop_last=False)
        rmse_total = 0
        r2_total = 0
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            mean_out,std_out=self.predict(x,y_normalizer)
            mean_out=torch.cat(mean_out,dim=0)
            std_out = torch.cat(std_out, dim=0)
            out = mean_out.view(batch_size, -1)
            y = y.view(batch_size, -1)
            out = out.cpu().detach()
            y = y.cpu().detach().numpy()
            rmse = cal_rmse(out, y)
            r2 = r2_score(out, y)
            rmse_total += rmse
            r2_total += r2
        rmse_total /= 10
        r2_total /= 10
        print("rmse:", rmse)
        print("r2:", r2)

    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def predict(self,x,y_normalizer,samples=20):
        pred_mean = []
        pred_std=[]
        self.enable_dropout()
            # 50 85 85 3
        for i in range(x.shape[0]):
            test_x=x[i,...]
            # print("test_x",test_x.shape)# 85 85 3
            test_x=test_x.unsqueeze(0) # 1 85 85 3
            result=[]
            for t in range(samples):
                out = self(test_x).reshape(test_x.shape[0],test_x.shape[1],test_x.shape[2])
                out = y_normalizer.inverse(out)
                out = out.cpu().detach()
                result.append(out)
            stacked_tensor=torch.stack(result,axis=0)
            mean=torch.mean(stacked_tensor,axis=0)
            std=torch.std(stacked_tensor,axis=0)
            pred_mean.append(mean)
            pred_std.append(std)
        #
        # print("region",len(pred_mean),len(pred_std)) # 500 500
        return pred_mean,pred_std
    def update_model(self,x_train,y_train,x_normalizer,y_normalizer):
        step_size = 100
        gamma = 0.5
        epochs=100
        batch_size = 20
        learning_rate = 0.005

        optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        x_train = x_normalizer.forward(x_train)
        y_train = y_normalizer.forward(y_train)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                                   batch_size=batch_size,
                                                   shuffle=True, drop_last=True)


        myloss = LpLoss(size_average=False)
        # train_loss = []
        # test_loss = []
        # epoch = []
        y_normalizer.cuda()
        for ep in range(epochs):
            # t1 = default_timer()
            train_l2 = 0
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                # out = model(x).reshape(batch_size, s, s)
                x = x.to(dtype=torch.double)
                out = self(x).reshape(batch_size, x.shape[1], x.shape[2])
                out = y_normalizer.inverse(out)
                y = y_normalizer.inverse(y)

                loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
                loss.backward()

                optimizer.step()
                train_l2 += loss.item()

            scheduler.step()

            # train_l2 /= x_train.shape[0]
            # t2 = default_timer()
            # print(ep, t2 - t1, train_l2)


################################################################
# configs
################################################################

