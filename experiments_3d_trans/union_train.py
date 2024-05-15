import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import scipy
import os

import sys
from fourier_mcdropout.fourier_union_4d import aro_4d
################################################################
# configs
###############################################################
###############################################
x_train_low = '../ev6_trans/input_mid.mat'
x_test_low= '../ev6_trans/input_mid.mat'
y_train_low = '../ev6_trans/output_mid.mat'
y_test_low= '../ev6_trans/output_mid.mat'
###############################################
x_train_high = '../ev6_trans/input_high.mat'
x_test_high= '../ev6_trans/input_high.mat'
y_train_high = '../ev6_trans/output_high.mat'
y_test_high= '../ev6_trans/output_high.mat'
import re
from fourier_mcdropout.utilities3 import setup_seed
import mat73
for arg in sys.argv:
    if 'low' in arg:
        ntrain_low =int(re.split('=',arg)[1])
    if 'high' in arg:
        ntrain_high =int(re.split('=',arg)[1])
    if 'seed' in arg:
        seed = int(re.split('=', arg)[1])
    if 'epoch' in arg:  # med high
        ep = int(re.split('=', arg)[1])
    if 'batch_size' in arg:  # med high
        bsize = int(re.split('=', arg)[1])
setup_seed(seed)
ntest= 500
#######
########

from fourier_mcdropout.utilities3 import *

x_train_low=mat73.loadmat(x_train_low)
x_train_low_data=x_train_low['data']
x_train_low=x_train_low_data[:ntrain_low,...]
x_test_low=x_train_low_data[-ntest:,...]
x_train_low=torch.tensor(x_train_low).to("cuda")
x_test_low=torch.tensor(x_test_low).to("cuda")


y_train_low =mat73.loadmat(y_train_low)
y_train_low_data=y_train_low['data']
y_train_low=y_train_low_data[:ntrain_low,...]
y_test_low=y_train_low_data[-ntest:,...]
y_train_low=torch.tensor(y_train_low).to("cuda")
y_test_low=torch.tensor(y_test_low).to("cuda")

### high
x_train_high=mat73.loadmat(x_train_high)
x_train_high_data= x_train_high['data']
x_train_high=x_train_high_data[:ntrain_high,...]
x_test_high=x_train_high_data[-ntest:,...]

x_train_high=torch.tensor(x_train_high).to("cuda")
x_test_high=torch.tensor(x_test_high).to("cuda")


y_train_data = mat73.loadmat(y_train_high)
y_train_high=y_train_data['data'][:ntrain_high,...]
y_train_high=torch.tensor(y_train_high).to("cuda")
y_test_high=y_train_data['data'][-ntest:,...]
y_test_high=torch.tensor(y_test_high).to("cuda")

# loader
print("low_samples:",x_train_low.shape)
print("high_samples:",x_train_high.shape)
################################################################
fno_set=[12, 12,2,2, 32]

setting=[ntrain_low,ntrain_high,ep,bsize,seed]

datalow=[x_train_low,y_train_low,x_test_low,y_test_low]

datahigh=[x_train_high,y_train_high,x_test_high,y_test_high]
# #
model=aro_4d(fno_set,fno_set).to("cuda")
#
model_low,x_normalizer_low,y_normalizer_low,folder=model.training_low(datalow,setting)
data_res=model.cal_res_dataset(model_low,datahigh,x_normalizer_low,y_normalizer_low)
model_res, x_normalizer_res, y_normalizer_res,folder=model.training_res(data_res,setting)
model.test_model(x_test_high,y_test_high,model_low,model_res,x_normalizer_low,y_normalizer_low,x_normalizer_res,y_normalizer_res)
pt_path=os.path.join(folder,'model.pt')
torch.save([model,x_normalizer_low,model_low,y_normalizer_low,x_normalizer_res,model_res,y_normalizer_res], pt_path)
#aro.test_model(x_test_high,y_test_high,model_low,model_high,x_norm_low,y_norm_low,x_norm_high,y_norm_high)