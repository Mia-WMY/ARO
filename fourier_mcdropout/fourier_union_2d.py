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
from fourier_mcdropout.normalize import normalize,cal_rmse,CombinedDataset
from sklearn.metrics import mean_squared_error,r2_score
torch.manual_seed(0)
np.random.seed(0)
from torch.utils.data import Dataset, DataLoader
from fourier_mcdropout.fourier_single_2d import FNO2d
################################################################
# z_fourier layer
################################################################
#######################################################

class aro_2d(nn.Module):
    def __init__(self, fno_low,fno_res):
        super(aro_2d, self).__init__()
        modes1_low, modes2_low,  width_low=fno_low
        modes1_res, modes2_res,  width_res=fno_res
        self.modes1_low=modes1_low
        self.modes2_low=modes2_low
        self.width_low=width_low
        self.modes1_res=modes1_res
        self.modes2_res=modes2_res
        self.width_res=width_res
        self.s1=151
        self.s2=151

    def training_low(self,data_low,settings):
        x_train_low, y_train_low, x_test_low, y_test_low = data_low
        ntrain_low,ntest_low,ntrain_high,ntest_high=settings
        # x_normalizer_low = normalize(x_train_low)
        # x_train_low = x_normalizer_low.forward(x_train_low)
        # x_test_low = x_normalizer_low.forward(x_test_low)
        # y_normalizer_low = normalize(y_train_low)
        # y_train_low = y_normalizer_low.forward(y_train_low)
        model = FNO2d(self.modes1_low, self.modes1_low, self.width_low).cuda()
        x_normalizer_low, y_normalizer_low, folder = model.train_model(x_train_low, x_test_low, y_train_low, y_test_low, ntrain_low, ntest_low,
                                                               work_dir=f'union_low{ntrain_low}_high{ntrain_high}')
        return model,x_normalizer_low,y_normalizer_low,folder

    def cal_res_dataset(self,model_low,data_high,x_normalizer_low,y_normalizer_low):
        x_train_high, y_train_high, x_test_high, y_test_high = data_high
        x_train_high=x_train_high.to('cuda')
        y_train_high=y_train_high.to('cuda')
        x_test_high=x_test_high.to('cuda')
        y_test_high=y_test_high.to('cuda')
        batch_size=20
        x_train_high = x_normalizer_low.forward(x_train_high)
        y_train_high = y_normalizer_low.forward(y_train_high)
        ####
        x_test_high = x_normalizer_low.forward(x_test_high)
        y_test_high = y_normalizer_low.forward(y_test_high)

        test_loader_high = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_high, y_test_high),
                                                       batch_size=batch_size,
                                                       shuffle=True, drop_last=True)
        train_loader_high = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_high, y_train_high),
                                                       batch_size=batch_size,
                                                       shuffle=True, drop_last=True)

        ####### cal_res
        all_outputs = []
        all_x = []
        with torch.no_grad():
            for x, y in train_loader_high:
                x, y = x.cuda(), y.cuda()
                out = model_low(x)
                out = y_normalizer_low.inverse(out)
                y = y_normalizer_low.inverse(y)
                x = x_normalizer_low.inverse(x)
                out = out[:, :, :, 0]
                all_outputs.append(y - out)
                all_x.append(x)
        res_y = torch.cat(all_outputs, dim=0)
        res_x = torch.cat(all_x, dim=0)

        ####
        all_outputs = []
        all_x = []
        with torch.no_grad():
            for x, y in test_loader_high:
                x, y = x.cuda(), y.cuda()
                out = model_low(x)
                out = y_normalizer_low.inverse(out)
                y = y_normalizer_low.inverse(y)
                x = x_normalizer_low.inverse(x)
                out = out[:, :, :, 0]
                all_outputs.append(y - out)
                all_x.append(x)
        res_ty = torch.cat(all_outputs, dim=0)
        res_tx = torch.cat(all_x, dim=0)
        data_res=[res_x,res_y,res_tx,res_ty]
        return data_res

    def training_res(self,data_res,settings):
        res_x,res_y,res_tx,res_ty=data_res
        # x_normalizer_res = normalize(res_x)
        # res_x = x_normalizer_res.forward(res_x)
        # res_tx = x_normalizer_res.forward(res_tx)
        # y_normalizer_res = normalize(res_y)
        # res_y = y_normalizer_res.forward(res_y)
        ntrain_low, ntest_low, ntrain_high, ntest_high = settings
        model = FNO2d(self.modes1_res, self.modes2_res, self.width_res).cuda()
        x_normalizer_res, y_normalizer_res, folder = model.train_model(res_x, res_tx, res_y, res_ty,
                                                                       ntrain_high, ntest_high,
                                                                       work_dir=f'union_low{ntrain_low}_high{ntrain_high}')
        return model, x_normalizer_res, y_normalizer_res,folder


    def test_model(self,x_test,y_test,model_low,model_res,x_normalizer_low,y_normalizer_low,x_normalizer_high,y_normalizer_high):
        x_test_low = x_normalizer_low.forward(x_test)
        x_test_high = x_normalizer_high.forward(x_test)
        combined_dataset = CombinedDataset(x_test_low, x_test_high, y_test)
        batch_size = 100
        data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        ################################################################
        total_r2 = 0
        total_rmse = 0
        for batch in data_loader:
            x, y, z = batch
            x_low, x_high, y = x.cuda(), y.cuda(), z.cuda()
            out_low_mean,out_low_std = model_low.predict(x_low,y_normalizer_low)
            out_high_mean,out_high_std=model_res.predict(x_high,y_normalizer_high)
            out_mean= [ a+b for a,b in zip(out_low_mean, out_high_mean)]
            out_std= [ a+b for a,b in zip(out_low_std,out_high_std)]
            out_mean = torch.cat(out_mean, dim=0)
            out_std = torch.cat(out_std,dim=0),
            out = out_mean.view(batch_size, -1)
            y = y.view(batch_size, -1)
            out = out.cpu().detach()
            y = y.cpu().detach().numpy()
            rmse = cal_rmse(out, y)
            r2 = r2_score(out, y)
            total_r2 += r2
            total_rmse += rmse
        print("rmse:", total_rmse / 10)
        print("r2:", total_r2 / 10)

    def model_predict(self,x_test,model_list):
        x_normalizer_low,model_low,y_normalizer_low,x_normalizer_high,model_res,y_normalizer_high=model_list

        x_test_low = x_normalizer_low.forward(x_test)
        x_test_high = x_normalizer_high.forward(x_test)
        x_low, x_high = x_test_low.cuda(), x_test_high.cuda()
        # print("x",x_low.shape)# 500 85 85 3
        out_low_mean, out_low_std = model_low.predict(x_low, y_normalizer_low)
        # print("low",len(out_low_mean),len(out_low_std
        #                                   ))
        # print("high",x_high.shape)
        out_high_mean, out_high_std = model_res.predict(x_high, y_normalizer_high)
        # print("high",len(out_high_mean),len(out_high_std))
        out_mean = [a + b for a, b in zip(out_low_mean, out_high_mean)]
        out_std = [a + b for a, b in zip(out_low_std, out_high_std)]
        # print(".",len(out_mean),len(out_std))
        out_mean = torch.cat(out_mean, dim=0)
        out_std = torch.cat(out_std, dim=0)
        return out_mean,out_std


    def update_low(self,model_low,x_low,y_low,x_norm,y_norm):
        model_low.update_model(x_low,  y_low, x_norm,y_norm)
        return model_low

    def update_res(self, model_low, x_low, y_low, x_norm, y_norm):
        model_low.update_model(x_low, y_low, x_norm, y_norm)
        return model_low

    def update_data(self,model_low,x_high,y_high, x_normalizer_low, y_normalizer_low):

        x_high = x_high.to('cuda')
        y_high = y_high.to('cuda')

        batch_size = 20
        x_high = x_normalizer_low.forward(x_high)
        y_high = y_normalizer_low.forward(y_high)
        ####


        train_loader_high = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_high, y_high),
                                                       batch_size=batch_size,
                                                       shuffle=True, drop_last=True)

        ####### cal_res
        all_outputs = []
        all_x = []
        with torch.no_grad():
            for x, y in train_loader_high:
                x, y = x.cuda(), y.cuda()
                out = model_low(x)
                out = y_normalizer_low.inverse(out)
                y = y_normalizer_low.inverse(y)
                x = x_normalizer_low.inverse(x)
                out = out[:, :, :, 0]
                all_outputs.append(y - out)
                all_x.append(x)
        res_y = torch.cat(all_outputs, dim=0)
        res_x = torch.cat(all_x, dim=0)

        ####

        return res_x,res_y


