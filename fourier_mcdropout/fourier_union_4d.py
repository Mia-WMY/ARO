
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
from fourier_mcdropout.fourier_4d import FNO4d
from fourier_mcdropout.normalize import normalize,cal_rmse,CombinedDataset
from torch.utils.data import Dataset, DataLoader


def sum_std_from_tensors(std_list_a, std_list_b):
    # 将列表转换为 PyTorch 的张量
    std_list_a = torch.tensor([item.cpu().detach().numpy() for item in std_list_a]).to("cuda")
    std_list_b = torch.tensor([item.cpu().detach().numpy() for item in std_list_b]).to("cuda")
    std_tensor_a = torch.tensor(std_list_a).to("cuda")
    std_tensor_b = torch.tensor(std_list_b).to("cuda")
    std_tensor_a=torch.squeeze(std_tensor_a).to("cuda")
    std_tensor_b=torch.squeeze(std_tensor_b).to("cuda")

    # 计算标准差之和
    sum_std = torch.sqrt(std_tensor_a ** 2+ std_tensor_b ** 2).to("cuda")

    return sum_std
class aro_4d(nn.Module):
    def __init__(self, fno_low,fno_res):
        super(aro_4d, self).__init__()
        modes1_low, modes2_low, modes3_low,modes4_low,width_low = fno_low
        modes1_res, modes2_res, modes3_res,modes4_res,width_res = fno_res
        self.modes1_low = modes1_low
        self.modes2_low = modes2_low
        self.modes3_low=modes3_low
        self.modes4_low=modes4_low
        self.width_low = width_low
        self.modes1_res = modes1_res
        self.modes2_res = modes2_res
        self.modes3_res=modes3_res
        self.modes4_res=modes4_res
        self.width_res = width_res

    def training_low(self,data_low,settings):
        x_train_low, y_train_low = data_low
        ntrain_low,ntrain_high,ep,bsize,seed=settings

        model = FNO4d(self.modes1_low, self.modes2_low,self.modes3_low, self.modes4_low,self.width_low).cuda()
        x_normalizer_low, y_normalizer_low, folder = model.train_model(x_train_low,y_train_low,epochs=ep,batch_size=bsize,
                                                               work_dir=f'union_low{ntrain_low}_high{ntrain_high}_{seed}')
        return model,x_normalizer_low,y_normalizer_low,folder

    def cal_res_dataset(self,model_low,data_high,x_normalizer_low,y_normalizer_low):
        x_train_high, y_train_high = data_high
        x_train_high=x_train_high.to('cuda')
        y_train_high=y_train_high.to('cuda')

        batch_size=10
        x_train_high = x_normalizer_low.forward(x_train_high)
        y_train_high = y_normalizer_low.forward(y_train_high)
        ####

        # test_loader_high = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_high, y_test_high),
        #                                                batch_size=batch_size,
        #                                                shuffle=True, drop_last=True)
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
                out = torch.squeeze(out)
                all_outputs.append(y - out)
                # print(y-out)
                all_x.append(x)
        res_y = torch.cat(all_outputs, dim=0)
        res_x = torch.cat(all_x, dim=0)
        data_res=[res_x,res_y]
        return data_res

    def valid_model(self, x_test, y_test, model_low, model_res, x_normalizer_low, y_normalizer_low, x_normalizer_high,
                    y_normalizer_high):
        x_test_low = x_normalizer_low.forward(x_test)
        x_test_high = x_normalizer_high.forward(x_test)
        combined_dataset = CombinedDataset(x_test_low, x_test_high, y_test)
        batch_size = 10
        data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
        ################################################################
        total_rmse = 0
        total_r2 = 0
        layer1_differ = 0
        layer2_differ = 0
        iter = 0
        for batch in data_loader:
            x, y, z = batch
            x_low, x_high, y = x.cuda(), y.cuda(), z.cuda()
            out_low_mean, out_low_std = model_low.predict(x_low, y_normalizer_low)
            out_high_mean, out_high_std = model_res.predict(x_high, y_normalizer_high)
            out_low_mean = torch.cat([tensor.clone().detach() for tensor in out_low_mean], dim=0)
            out_high_mean = torch.cat([tensor.clone().detach() for tensor in out_high_mean], dim=0)
            y = y.cpu().detach().numpy()
            out = out_low_mean + out_high_mean
            for i in range(x.shape[0]):
                l = 0
                differ1 = torch.mean(torch.abs(out[i, ..., l] - y[i, ..., l]))
                l = 1
                differ2 = torch.mean(torch.abs(out[i, ..., l] - y[i, ..., l]))
                r2 = 0
                rmse = 0
                for l in range(y.shape[-1]):
                    r2 += r2_score(out[i, ..., l].flatten(), y[i, ..., l].flatten())
                    rmse += cal_rmse(out[i, ..., l], y[i, ..., l])
                total_r2 += r2 / y.shape[-1]
                total_rmse += rmse / y.shape[-1]
                layer1_differ += differ1
                layer2_differ += differ2
                iter += 1

        print("rmse:", total_rmse / iter)
        print("r2", total_r2 / iter)
        print("layer1 terr", layer1_differ / iter)
        print("layer2 terr", layer2_differ / iter)
    def speed_model(self, x_test, y_test, model_low, model_res, x_normalizer_low, y_normalizer_low, x_normalizer_high,
                    y_normalizer_high):
        x_test_low = x_normalizer_low.forward(x_test)
        x_test_high = x_normalizer_high.forward(x_test)
        combined_dataset = CombinedDataset(x_test_low, x_test_high, y_test)
        batch_size = 10
        data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        total_time = 0
        for batch in data_loader:
            x, y, z = batch
            x_low, x_high, y = x.cuda(), y.cuda(), z.cuda()
            t1 = default_timer()
            out_low_mean, out_low_std = model_low.predict(x_low, y_normalizer_low)
            out_high_mean, out_high_std = model_res.predict(x_high, y_normalizer_high)
            out_low_mean = torch.cat([tensor.clone().detach() for tensor in out_low_mean], dim=0)
            out_high_mean = torch.cat([tensor.clone().detach() for tensor in out_high_mean], dim=0)
            y = y.cpu().detach().numpy()
            out = out_low_mean + out_high_mean
            t2 = default_timer()
            total_time += t2 - t1
        print("infer time:", total_time / x_test.shape[0])

    def training_res(self,data_res,settings):
        res_x,res_y=data_res
        ntrain_low, ntrain_high, ep, bsize, seed = settings

        model = FNO4d(self.modes1_res, self.modes2_res, self.modes3_res,self.modes4_res,self.width_res).cuda()
        x_normalizer_res, y_normalizer_res, folder = model.train_model(res_x,  res_y, epochs=ep,batch_size=bsize,
                                                                       work_dir=f'union_low{ntrain_low}_high{ntrain_high}_{seed}')
        return model, x_normalizer_res, y_normalizer_res,folder


    def test_model(self,x_test,y_test,model_low,model_res,x_normalizer_low,y_normalizer_low,x_normalizer_high,y_normalizer_high):
        x_test_low = x_normalizer_low.forward(x_test)
        x_test_high = x_normalizer_high.forward(x_test)
        combined_dataset = CombinedDataset(x_test_low, x_test_high, y_test)
        batch_size = 10
        data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

        ################################################################
        rmse_total = 0
        r2_total = 0
        maxx_total = 0
        differ_total = 0
        mape_total = 0
        pape_total = 0
        iter=0
        for batch in data_loader:
            x, y, z = batch
            x_low, x_high, y = x.cuda(), y.cuda(), z.cuda()
            out_low_mean,out_low_std = model_low.predict(x_low,y_normalizer_low)
            out_high_mean,out_high_std=model_res.predict(x_high,y_normalizer_high)
            out_low_mean = torch.cat([tensor.clone().detach() for tensor in out_low_mean], dim=0)
            out_high_mean = torch.cat([tensor.clone().detach() for tensor in out_high_mean], dim=0)
            y = y.cpu().detach().numpy()
            out = out_low_mean + out_high_mean
            for i in range(x.shape[0]):
                rmse = 0
                maxx=0
                differ = 0
                mape = 0
                pape = 0
                for l in range(y.shape[-1]):
                    rmse += cal_rmse(out[i, ..., l], y[i, ..., l])
                    maxx += torch.max(torch.abs(out[i, ..., l] - y[i, ..., l]))
                    differ += torch.mean(torch.abs(out[i, ..., l] - y[i, ..., l]))
                    mape += torch.mean(torch.abs(out[i, ..., l] - y[i, ..., l]) / y[i, ..., l]) * 100
                    pape += torch.max(torch.abs(out[i, ..., l] - y[i, ..., l]) / y[i, ..., l]) * 100
                rmse = rmse / y.shape[-1]
                maxx = maxx / y.shape[-1]
                differ = differ / y.shape[-1]
                mape = mape / y.shape[-1]
                pape = pape / y.shape[-1]
                r2 = 0
                for l in range(y.shape[-1]):
                    r2 += r2_score(out[i, ..., l].flatten(), y[i, ..., l].flatten())
                r2 = r2 / y.shape[-1]

                r2 = r2 / y.shape[-1]
                rmse_total += rmse
                r2_total += r2
                maxx_total += maxx
                differ_total += differ
                mape_total += mape
                pape_total += pape
                iter += 1
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
        out_std = sum_std_from_tensors(out_low_std,out_high_std)
        # print(".",len(out_mean),len(out_std))
        out_mean = torch.cat(out_mean, dim=0)

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

        batch_size = 10
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
                out = out[..., 0]
                all_outputs.append(y - out)
                all_x.append(x)
        res_y = torch.cat(all_outputs, dim=0)
        res_x = torch.cat(all_x, dim=0)

        ####

        return res_x,res_y





