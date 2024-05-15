from timeit import default_timer

import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
default_dtype = np.float64
import matplotlib.pyplot as plt
import scipy
import os
torch.manual_seed(0)
np.random.seed(0)
import sys
from fourier_mcdropout.fourier_union_3d import aro_3d
################################################################
# configs
###############################################################
###############################################
x_train_low = '../3d_steady/input_mid.mat'
x_test_low= '../3d_steady/input_mid.mat'
y_train_low = '../3d_steady/output_mid.mat'
y_test_low= '../3d_steady/output_mid.mat'
###############################################
x_train_high = '../3d_steady/input_high.mat'
x_test_high= '../3d_steady/input_high.mat'
y_train_high = '../3d_steady/output_high.mat'
y_test_high= '../3d_steady/output_high.mat'

import re
from fourier_mcdropout.utilities3 import setup_seed
for arg in sys.argv:
    if 'low_num' in arg:
        ntrain_low =int(re.split('=',arg)[1])
    if 'high_num' in arg:
        ntrain_high =int(re.split('=',arg)[1])
    if 'seed' in arg:
        seed = int(re.split('=', arg)[1])
    if 'epoch' in arg: # med high
        ep = int(re.split('=', arg)[1])
    if 'batch_size' in arg:  # med high
        bsize= int(re.split('=', arg)[1])

setup_seed(seed)
from fourier_mcdropout.utilities3 import *
import mat73
ntest=500
x_train_low=mat73.loadmat(x_train_low)
x_train_low=x_train_low['data'].astype(np.float32)
x_train_low_data=torch.tensor(x_train_low)
x_train_low=x_train_low_data[:ntrain_low,...]
x_test_low=x_train_low_data[-ntest:,...]
y_train_low =mat73.loadmat(y_train_low)
y_train_low=y_train_low['data'].astype(np.float32)
y_train_low_data=torch.tensor(y_train_low)
y_train_low=y_train_low_data[:ntrain_low,...]
y_test_low=y_train_low_data[-ntest:,...].to(torch.float32)
x_train_low=x_train_low.to('cuda')
x_test_low=x_test_low.to('cuda')

### high
x_train_high=mat73.loadmat(x_train_high)
x_train_high=x_train_high['data'].astype(np.float32)
x_train_high_data=torch.tensor(x_train_high)
x_train_high=x_train_high_data[:ntrain_high,...].to(torch.float32)
x_test_high=x_train_high_data[-ntest:,...].to(torch.float32)
y_train_high = mat73.loadmat(y_train_high)
y_train_high =y_train_high['data'].astype(np.float32)
y_train_high_data=torch.tensor(y_train_high)
y_train_high=y_train_high_data[:ntrain_high,...].to(torch.float32)
y_test_high=y_train_high_data[-ntest:,...].to(torch.float32)
x_train_high=x_train_high.to('cuda')
x_test_high=x_test_high.to('cuda')
# loader
print("low_samples:",x_train_low.shape)
print("high_samples:",x_train_high.shape)
print("testx_samples:",x_test_high.shape)
################################################################
fno_set=[12, 12,2, 32]

setting=[ntrain_low,ntrain_high,ep,bsize,seed]

datalow=[x_train_low,y_train_low,x_test_low,y_test_low]

datahigh=[x_train_high,y_train_high,x_test_high,y_test_high]
# #
model=aro_3d(fno_set,fno_set)
#
t1 = default_timer()
model_low,x_normalizer_low,y_normalizer_low,folder=model.training_low(datalow,setting)
data_res=model.cal_res_dataset(model_low,datahigh,x_normalizer_low,y_normalizer_low)
model_res, x_normalizer_res, y_normalizer_res,folder=model.training_res(data_res,setting)
t2=default_timer()
print("total_time",t2-t1)
model.test_model(x_test_high,y_test_high,model_low,model_res,x_normalizer_low,y_normalizer_low,x_normalizer_res,y_normalizer_res)
pt_path=os.path.join(folder,'model.pt')
torch.save([model,x_normalizer_low,model_low,y_normalizer_low,x_normalizer_res,model_res,y_normalizer_res], pt_path)

