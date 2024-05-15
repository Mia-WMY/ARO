import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
default_dtype = np.float64
import sys
import os
import scipy
import warnings
def test_model(models,x_test_high,y_test_high):
    aro = models[0]
    x_norm_low = models[1]
    model_low = models[2]
    y_norm_low = models[3]
    x_norm_high = models[4]
    model_high = models[5]
    y_norm_high = models[6]
    aro.test_model(x_test_high, y_test_high, model_low, model_high, x_norm_low, y_norm_low, x_norm_high, y_norm_high)

def retrain_with_update(model,selected_x_low,selected_x_high,selected_y_low,selected_y_high,x_train_low,y_train_low,x_train_high,y_train_high):
    ############################################################
    # configs
    ###############################################################
    ##
    # update data
    update_low=True
    update_high = True
    if selected_x_low.numel() !=0:
        x_train_low=torch.cat((x_train_low.to("cuda"),selected_x_low.to("cuda")),dim=0)
        y_train_low=torch.cat((y_train_low.to("cuda"),selected_y_low.to("cuda")),dim=0)

    else:
        update_low=False
    if selected_x_high.numel() !=0:
        x_train_high=torch.cat((x_train_high.to('cuda'),selected_x_high.to("cuda")),dim=0)
        y_train_high=torch.cat((y_train_high.to("cuda"),selected_y_high.to("cuda")),dim=0)
    else:
        update_high=False
    #
    aro=model[0]
    x_low_norm=model[1]
    model_low=model[2]
    y_low_norm=model[3]
    x_res_norm=model[4]
    model_res=model[5]
    y_res_norm=model[6]
    # update model
    if update_low:
        model_low=aro.update_low(model_low,x_train_low,y_train_low,x_low_norm,y_low_norm)
    if update_high:
        res_x,res_y = aro.update_data(model_low,x_train_high,y_train_high, x_low_norm, y_low_norm)
        model_res = aro.update_res(model_res,res_x,res_y,x_res_norm,y_res_norm)
    new_models=[aro,x_low_norm,model_low,y_low_norm,x_res_norm,model_res,y_res_norm]

    return x_train_low,y_train_low,x_train_high,y_train_high,new_models

def remove_indices(tensor,indices_to_move):

    indices_to_keep=[i for i in range(tensor.size(0)) if i not in indices_to_move]
    filter_tensor=torch.index_select(tensor.to("cuda"),0,torch.tensor(indices_to_keep).to("cuda"))

    return filter_tensor


def remove_selected_pool(x_pool,y_pool,queries_fid,queries_index):
    low_index=[]
    high_index=[]
    for fid,index in zip(queries_fid,queries_index):
        if fid == 0:
            low_index.append(index)
        else:
            high_index.append(index)
    print(f"select {len(low_index)} low fidelity, {len(high_index)} high fidelity :)")
    new_x_pool=[]
    new_y_pool=[]
    ## filter x_low y_low
    if len(low_index) !=0:

        filter_x_low=remove_indices(x_pool[0],low_index)
        filter_y_low=remove_indices(y_pool[0],low_index)
    else:
        filter_x_low=x_pool[0]
        filter_y_low=y_pool[0]
    if len(high_index) !=0:
        # filter_index=[i for i in range(x_pool[1]) if i not in high_index]
        # filter_x_high=x_pool[1][filter_index]
        # filter_y_high=y_pool[1][filter_index]
        filter_x_high=remove_indices(x_pool[1],high_index)
        filter_y_high=remove_indices(y_pool[1],high_index)
    else:
        filter_x_high=x_pool[1]
        filter_y_high=y_pool[1]
    new_x_pool.append(filter_x_low.to("cuda"))
    new_x_pool.append(filter_x_high.to("cuda"))
    new_y_pool.append(filter_y_low.to("cuda"))
    new_y_pool.append(filter_y_high.to("cuda"))
    return new_x_pool,new_y_pool


def select_tensors_by_fidelity(list_fidelity,list_index,list_x,list_y):
    tensor_low=[]
    tensor_high=[]
    tensor_y_low=[]
    tensor_y_high=[]
    list_x[0]=list_x[0].to("cuda")
    list_x[1]=list_x[1].to("cuda")
    list_y[0] = list_y[0].to("cuda")
    list_y[1] = list_y[1].to("cuda")
    for fildeity,index in zip(list_fidelity,list_index):

        if fildeity ==0:
            selected_low=torch.index_select(list_x[0],0,torch.tensor(index).to("cuda"))
            tensor_low.append(selected_low)
            selected_low = torch.index_select(list_y[0], 0, torch.tensor(index).to("cuda"))
            tensor_y_low.append(selected_low)
        else:
            selected_high=torch.index_select(list_x[1],0,torch.tensor(index).to("cuda"))
            tensor_high.append(selected_high)
            selected_high = torch.index_select(list_y[1], 0, torch.tensor(index).to("cuda"))
            tensor_y_high.append(selected_high)
    if len(tensor_low) !=0:
        concated_low=torch.cat(tensor_low,dim=0)
        concated_y_low = torch.cat(tensor_y_low, dim=0)
    else:
        concated_low=torch.tensor([])
        concated_y_low=torch.tensor([])
    if len(tensor_high) !=0:
        concated_high=torch.cat(tensor_high,dim=0)
        concated_y_high=torch.cat(tensor_y_high,dim=0)
    else:
        concated_high=torch.tensor([])
        concated_y_high=torch.tensor([])
    # print("selected",concated_low.shape,concated_high.shape)
    return concated_low,concated_high,concated_y_low,concated_y_high

def get_mf_preds(x_pool,aro_model,model_list):
    pred_mu_samples_list = []
    pred_std_samples_list = []
    for x in x_pool:
        sf_data_loader = torch.utils.data.DataLoader(
            x,
            batch_size=50,
            shuffle=False,
        )

        sf_pred_mu_samples = []
        sf_pred_std_samples = []
                # print(len(sf_pred_mu_samples))
        for x in sf_data_loader: # 50 85 85 3
            # x=x.squeeze(0) # 50 85 85 3
            out_mean, out_std = aro_model.model_predict(x,model_list)
            sf_pred_mu_samples.append(out_mean)
            sf_pred_std_samples.append(out_std)
            # print(len(sf_pred_mu_samples))
        index=0
        for x in sf_pred_mu_samples:
            if index == 0:
                sf_pred_mu=x
            else:
                sf_pred_mu=torch.cat((sf_pred_mu,x),dim=0)
            index+=1

        index=0
        for x in sf_pred_std_samples:
            if index == 0:
                sf_pred_std=x
            else:
                sf_pred_std=torch.cat((sf_pred_std,x),dim=0)
            index+=1
        #     index+=1
        # print(sf_pred_mu.shape,sf_pred_std.shape)

        pred_mu_samples_list.append(sf_pred_mu)
        pred_std_samples_list.append(sf_pred_std)

    return pred_mu_samples_list,pred_std_samples_list
def get_mf_x_pool(ntrain_low,ntrain_high,pool_size):
    x_pool=[]
    y_pool=[]
    x_train_low = '../aro_data/core8_steady/mid_x_train.npy'
    y_train_low = '../aro_data/core8_steady/mid_y_train.npy'
    ######################
    x_train_high = '../aro_data/core8_steady/high_x_train.npy'
    y_train_high = '../aro_data/core8_steady/high_y_train.npy'
    #####
    ################################################################
    ### low
    x_train_low = np.load(x_train_low)
    x_train_low = torch.tensor(x_train_low)
    x_pool_low = x_train_low[ntrain_low:ntrain_low+pool_size, ...].to('cuda')
    x_al_low = x_train_low[:ntrain_low, ...].to('cuda')
    x_re_low=x_train_low[ntrain_low:ntrain_low+800, ...].to('cuda')
    y_train_low =np.load(y_train_low)
    y_train_low = torch.tensor(y_train_low)
    y_pool_low = y_train_low[ntrain_low:ntrain_low + pool_size, ...]
    y_pool_low = y_pool_low.to('cuda')
    y_al_low = y_train_low[:ntrain_low , ...]
    y_re_low = y_train_low[ntrain_low:ntrain_low+800, ...].to('cuda')
    y_al_low = y_al_low.to('cuda')
    ### high
    x_train_high =np.load(x_train_high)
    x_train_high = torch.tensor(x_train_high)
    x_pool_high = x_train_high[ntrain_high:ntrain_high+pool_size, ...].to('cuda')
    x_al_high = x_train_high[:ntrain_high, ...]
    x_re_high = x_train_high[ntrain_high:ntrain_high+500, ...]
    y_train_high = np.load(y_train_high)
    y_train_high = torch.tensor(y_train_high)
    y_pool_high = y_train_high[ntrain_high:ntrain_high + pool_size, ...]
    y_pool_high = y_pool_high.to('cuda')
    y_al_high = y_train_high[:ntrain_high, ...]
    y_re_high = y_train_high[ntrain_high:ntrain_high+500 , ...]
    y_al_high = y_al_high.to('cuda')
    x_pool.append(x_pool_low)
    x_pool.append(x_pool_high)
    y_pool.append(y_pool_low)
    y_pool.append(y_pool_high)
    x_te_pool=[]
    y_te_pool=[]
    x_te_pool.append(x_re_low)
    x_te_pool.append(x_re_high)
    y_te_pool.append(y_re_low)
    y_te_pool.append(y_re_high)
    return x_pool,y_pool,x_te_pool,y_te_pool,x_al_low,y_al_low,x_al_high,y_al_high

def select_and_remove_from_pools(x, y, t):
    # 转换为 NumPy 数组
    np_x = np.array(x.cpu().detach())
    np_y = np.array(y.cpu().detach())

    # 获取张量的形状
    shape_x = np_x.shape
    shape_y = np_y.shape

    # 获取第一维度的长度
    num_tensors = shape_x[0]

    # 随机选择 t 个张量的索引
    selected_indices = np.random.choice(num_tensors, t, replace=False)

    # 选中的张量
    selected_tensors_x = np_x[selected_indices]
    selected_tensors_y = np_y[selected_indices]

    # 移除选中的张量后剩余的张量
    remaining_tensors_x = np.delete(np_x, selected_indices, axis=0)
    remaining_tensors_y = np.delete(np_y, selected_indices, axis=0)
    remaining_tensors_x=torch.tensor(remaining_tensors_x).to("cuda")
    remaining_tensors_y=torch.tensor(remaining_tensors_y).to("cuda")
    selected_tensors_x=torch.tensor(selected_tensors_x).to("cuda")
    selected_tensors_y = torch.tensor(selected_tensors_y).to("cuda")
    return selected_tensors_x, selected_tensors_y, remaining_tensors_x, remaining_tensors_y


def align_tensors(x_pool, y_pool,psize,x_te_pool,y_te_pool):
    x_all_pool=[]
    y_all_pool=[]
    x_new_pool=[]
    y_new_pool=[]
    x_re_pool = []
    y_re_pool=[]
    # random new pool
    for x,rest_x in zip (x_pool,x_te_pool):
        rest_all_x=torch.cat((x.to("cuda"),rest_x.to("cuda")),dim=0)
        x_all_pool.append(rest_all_x)
    for y,rest_y in zip (y_pool,y_te_pool):
        rest_all_y=torch.cat((y.to("cuda"),rest_y.to("cuda")),dim=0)
        y_all_pool.append(rest_all_y)
    attach_x, attach_y, rest_x, rest_y = select_and_remove_from_pools(x_all_pool[0], y_all_pool[0], psize)
    x_new_pool.append(attach_x)
    y_new_pool.append(attach_y)
    x_re_pool.append(rest_x)
    y_re_pool.append(rest_y)
    attach_x, attach_y, rest_x, rest_y = select_and_remove_from_pools(x_all_pool[1], y_all_pool[1], psize)
    x_new_pool.append(attach_x)
    y_new_pool.append(attach_y)
    x_re_pool.append(rest_x)
    y_re_pool.append(rest_y)
    return x_new_pool,y_new_pool,x_re_pool,y_re_pool


def select_and_remove_highsamples(x_pool, y_pool,x_te_pool,y_te_pool, num_select,psize):
    # 确定每次选择的样本数量不超出范围
    x_new_pool=[]
    y_new_pool=[]
    # 确定每次选择的样本数量不超出范围
    x_re_pool=[]
    y_re_pool=[]
    x_new_pool.append(x_pool[0])
    y_new_pool.append(y_pool[0])
    x_re_pool.append(x_te_pool[0])
    y_re_pool.append(y_te_pool[0])

    tensor1=x_pool[1]
    tensor2=y_pool[1]
    total_samples1 = tensor1.shape[0]
    total_samples2 = tensor2.shape[0]


    # 从张量1中随机选择 num_select 个样本
    selected_indices1 = torch.randperm(total_samples1)[:num_select]
    selected_samples1 = tensor1[selected_indices1]
    selected_samples2 = tensor2[selected_indices1]

    # 从原始张量中移除已选择的样本
    remaining_indices1 = [i for i in range(total_samples1) if i not in selected_indices1]
    remaining_samples1 = tensor1[remaining_indices1]

    remaining_indices2 = [i for i in range(total_samples2) if i not in selected_indices1]
    remaining_samples2 = tensor2[remaining_indices2]
    x_all_pool=[]
    y_all_pool=[]
    x_all_pool.append(x_pool[0])
    rest_all_x = torch.cat((remaining_samples1.to("cuda"), x_te_pool[1].to("cuda")), dim=0)
    x_all_pool.append(rest_all_x)
    y_all_pool.append(y_pool[0])
    rest_all_y = torch.cat((remaining_samples2.to("cuda"), y_te_pool[1].to("cuda")), dim=0)
    y_all_pool.append(rest_all_y)
    attach_x, attach_y, rest_x, rest_y = select_and_remove_from_pools(x_all_pool[1], y_all_pool[1], psize)
    x_new_pool.append(attach_x)
    y_new_pool.append(attach_y)
    x_re_pool.append(rest_x)
    y_re_pool.append(rest_y)

    return selected_samples1, selected_samples2, x_new_pool,y_new_pool,x_re_pool,y_re_pool


def select_and_remove_lowsamples(x_pool, y_pool,x_te_pool,y_te_pool, num_select,psize):
    # 确定每次选择的样本数量不超出范围
    x_new_pool=[]
    y_new_pool=[]
    # 确定每次选择的样本数量不超出范围
    x_re_pool=[]
    y_re_pool=[]

    tensor1=x_pool[0]
    tensor2=y_pool[0]
    total_samples1 = tensor1.shape[0]
    total_samples2 = tensor2.shape[0]


    # 从张量1中随机选择 num_select 个样本
    selected_indices1 = torch.randperm(total_samples1)[:num_select]
    selected_samples1 = tensor1[selected_indices1]
    selected_samples2 = tensor2[selected_indices1]

    # 从原始张量中移除已选择的样本
    remaining_indices1 = [i for i in range(total_samples1) if i not in selected_indices1]
    remaining_samples1 = tensor1[remaining_indices1]

    remaining_indices2 = [i for i in range(total_samples2) if i not in selected_indices1]
    remaining_samples2 = tensor2[remaining_indices2]
    x_all_pool=[]
    y_all_pool=[]
    # x_all_pool.append(x_pool[0])
    rest_all_x = torch.cat((remaining_samples1.to("cuda"), x_te_pool[0].to("cuda")), dim=0)
    x_all_pool.append(rest_all_x)
    # y_all_pool.append(y_pool[0])
    rest_all_y = torch.cat((remaining_samples2.to("cuda"), y_te_pool[0].to("cuda")), dim=0)
    y_all_pool.append(rest_all_y)
    attach_x, attach_y, rest_x, rest_y = select_and_remove_from_pools(x_all_pool[0], y_all_pool[0], psize)
    x_new_pool.append(attach_x)
    x_new_pool.append(x_pool[1])
    y_new_pool.append(attach_y)
    y_new_pool.append(y_pool[1])
    x_re_pool.append(rest_x)
    x_re_pool.append(x_te_pool[1])
    y_re_pool.append(rest_y)
    y_re_pool.append(y_te_pool[1])


    return selected_samples1, selected_samples2, x_new_pool,y_new_pool,x_re_pool,y_re_pool


def select_and_remove_ratiosamples(x_pool, y_pool,x_te_pool,y_te_pool,num_low, num_high,psize):
    # 确定每次选择的样本数量不超出范围
    x_new_pool=[]
    y_new_pool=[]
    # 确定每次选择的样本数量不超出范围
    x_re_pool=[]
    y_re_pool=[]

    tensor1=x_pool[0]
    tensor2=y_pool[0]
    total_samples1 = tensor1.shape[0]
    total_samples2 = tensor2.shape[0]

    indices=list(range(0,num_low))
    # 从张量1中随机选择 num_select 个样本
    selected_indices1 = torch.tensor(indices)
    selected_samples_xlow = tensor1[selected_indices1]
    selected_samples_ylow = tensor2[selected_indices1]

    # 从原始张量中移除已选择的样本
    remaining_indices1 = [i for i in range(total_samples1) if i not in selected_indices1]
    remaining_samples1 = tensor1[remaining_indices1]

    remaining_indices2 = [i for i in range(total_samples2) if i not in selected_indices1]
    remaining_samples2 = tensor2[remaining_indices2]
    x_all_pool=[]
    y_all_pool=[]
    # x_all_pool.append(x_pool[0])
    rest_all_x = torch.cat((remaining_samples1.to("cuda"), x_te_pool[0].to("cuda")), dim=0)
    x_all_pool.append(rest_all_x)
    # y_all_pool.append(y_pool[0])
    rest_all_y = torch.cat((remaining_samples2.to("cuda"), y_te_pool[0].to("cuda")), dim=0)
    y_all_pool.append(rest_all_y)
    attach_x, attach_y, rest_x, rest_y = select_and_remove_from_pools(x_all_pool[0], y_all_pool[0], psize)
    x_new_pool.append(attach_x)

    y_new_pool.append(attach_y)

    x_re_pool.append(rest_x)

    y_re_pool.append(rest_y)

    tensor1 = x_pool[1]
    tensor2 = y_pool[1]
    total_samples1 = tensor1.shape[0]
    total_samples2 = tensor2.shape[0]

    # 从张量1中随机选择 num_select 个样本
    indices = list(range(0, num_high))
    # 从张量1中随机选择 num_select 个样本
    selected_indices1 = torch.tensor(indices)
    selected_samples_xhigh = tensor1[selected_indices1]
    selected_samples_yhigh = tensor2[selected_indices1]

    # 从原始张量中移除已选择的样本
    remaining_indices1 = [i for i in range(total_samples1) if i not in selected_indices1]
    remaining_samples1 = tensor1[remaining_indices1]

    remaining_indices2 = [i for i in range(total_samples2) if i not in selected_indices1]
    remaining_samples2 = tensor2[remaining_indices2]

    # x_all_pool.append(x_pool[0])
    rest_all_x = torch.cat((remaining_samples1.to("cuda"), x_te_pool[1].to("cuda")), dim=0)
    x_all_pool.append(rest_all_x)

    rest_all_y = torch.cat((remaining_samples2.to("cuda"), y_te_pool[1].to("cuda")), dim=0)
    y_all_pool.append(rest_all_y)
    attach_x, attach_y, rest_x, rest_y = select_and_remove_from_pools(x_all_pool[1], y_all_pool[1], psize)
    x_new_pool.append(attach_x)

    y_new_pool.append(attach_y)

    x_re_pool.append(rest_x)
    y_re_pool.append(rest_y)


    return selected_samples_xlow, selected_samples_ylow,selected_samples_xhigh,selected_samples_yhigh, x_new_pool,y_new_pool,x_re_pool,y_re_pool

