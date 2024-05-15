import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fourier_mcdropout.fourier_4d import FNO4d

"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
default_dtype = np.float16
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer



from fourier_mcdropout.utilities3 import setup_seed
import re
################################################################
# configs
###############################################################
ntest = 500
for arg in sys.argv:
    if 'train_num' in arg:
        ntrain =int(re.split('=',arg)[1])
    if 'seed' in arg:
        seed = int(re.split('=', arg)[1])
    if 'fid' in arg: # med high
        fid = re.split('=', arg)[1]
    if 'epoch' in arg:  # med high
        ep = int(re.split('=', arg)[1])
    if 'batch_size' in arg:  # med high
        bsize = int(re.split('=', arg)[1])
setup_seed(seed)
xl_data = f'../core8_trans/input_{fid}.mat'
yl_data=f'../core8_trans/output_{fid}.mat'
xh_data = f'../core8_trans/input_high.mat'
yh_data=f'../core8_trans/output_high.mat'
import mat73
################################################################
# load data and data normalization
################################################################
from fourier_mcdropout.utilities3 import *
xl_data=mat73.loadmat(xl_data)
xl_data=xl_data['data']
xl_data=torch.tensor(xl_data).to("cuda")
x_train=xl_data[:ntrain,...]

yl_data = mat73.loadmat(yl_data)
yl_data = yl_data['data']
yl_data = torch.tensor(yl_data).to("cuda")
y_train=yl_data[:ntrain,...]
xh_data= mat73.loadmat(xh_data)
xh_data=xh_data['data']
xh_data=torch.tensor(xh_data).to("cuda")
x_test=xh_data[-ntest:,...].to(torch.float16)
yh_data = mat73.loadmat(yh_data)
yh_data=yh_data['data']
yh_data=torch.tensor(yh_data).to("cuda")
y_test=yh_data[-ntest:,...]
print("training samples:",x_train.shape)
print("testing samples:",x_test.shape)
model = FNO4d(modes1=12, modes2=12,modes3=2, modes4=2,width=32).cuda()
x_normalizer,y_normalizer,folder=model.train_model(x_train,y_train,ep,bsize,work_dir=f'single_trans_{fid}_{ntrain}_{seed}')
model.test_model(x_test,y_test,x_normalizer,y_normalizer)
pt_path=os.path.join(folder,'model.pt')
torch.save([x_normalizer,model,y_normalizer], pt_path)







