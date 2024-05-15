import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
default_dtype = np.float64
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import scipy
from fourier_mcdropout.mutual_info import mf_mutual_info
from active_learning import *

import re
from fourier_mcdropout.utilities3 import setup_seed
for arg in sys.argv:
    if 'low_num' in arg:
        low_num =int(re.split('=',arg)[1])
    if 'high_num' in arg:
        high_num =int(re.split('=',arg)[1])
    if 'seed' in arg:
        seed = int(re.split('=', arg)[1])
    if 'pool_size' in arg:
        psize = int(re.split('=', arg)[1])
    if 'budget' in arg:
        budget = int(re.split('=', arg)[1])
    if 'update' in arg:
        update = int(re.split('=', arg)[1])
x_asset = np.load("x_asset.npy")
x_asset=torch.tensor(x_asset).to('cuda')
y_asset = np.load("y_asset.npy")
y_asset=torch.tensor(y_asset).to('cuda')
x_pool,y_pool,x_te_pool,y_te_pool,x_train_low,y_train_low,x_train_high,y_train_high=get_mf_x_pool(ntrain_low=low_num,ntrain_high=high_num,pool_size=psize)
aro_path=f'union_low{low_num}_high{high_num}_1\model.pt'
selected_y_low =  torch.zeros(0)
selected_x_low = torch.zeros(0)
import os
for update_time in range(update):
    print(f"active learning update: {update_time}")
    models=torch.load(aro_path)# aro, x_low,model_low,y_low,x_res,model_res,y_res
    model_list=[models[1],models[2],models[3],models[4],models[5],models[6]]
    selected_x_high,selected_y_high,x_pool,y_pool,x_te_pool,y_te_pool=select_and_remove_highsamples(x_pool,y_pool,x_te_pool,y_te_pool,int(budget/4),psize)
    x_train_low,y_train_low,x_train_high,y_train_high,models=retrain_with_update(models,selected_x_low,selected_x_high,selected_y_low,selected_y_high,x_train_low,y_train_low,x_train_high,y_train_high)
    print("train low:", x_train_low.shape)
    print("train high:", x_train_high.shape)
    test_model(models,x_asset,y_asset)
    current_directory = os.getcwd()
    folder_name=f'union_low{low_num}_high{high_num}_1'
    folder_path = os.path.join(current_directory, folder_name)
    aro_path = os.path.join(folder_path, f'model_high{update_time}.pt')
    torch.save(models, aro_path)

