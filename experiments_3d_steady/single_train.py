import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fourier_mcdropout.fourier_3d import FNO3d

"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
default_dtype = np.float32


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
    if 'epoch' in arg: # med high
        ep = int(re.split('=', arg)[1])
    if 'batch_size' in arg:  # med high
        bsize= int(re.split('=', arg)[1])
x_train = f'../3d_steady/input_{fid}.mat'
y_train=f'../3d_steady/output_{fid}.mat'
x_test = f'../3d_steady/input_high.mat'
y_test=f'../3d_steady/output_high.mat'

################################################################
# load data and data normalization
################################################################
import os
from fourier_mcdropout.utilities3 import *
import mat73
x_train=mat73.loadmat(x_train)
x_train=x_train['data'].astype(np.float32)
x_train=torch.tensor(x_train)
x_train=x_train[:ntrain,...].to(torch.float32)
y_train = mat73.loadmat(y_train)
y_train=y_train['data'].astype(np.float32)
y_train=torch.tensor(y_train)
y_train=y_train[:ntrain,...].to(torch.float32)
x_test = mat73.loadmat(x_test)
x_test=x_test['data'].astype(np.float32)
x_test=torch.tensor(x_test)
x_test=x_test[-ntest:,...].to(torch.float32)
y_test = mat73.loadmat(y_test)
y_test=y_test['data'].astype(np.float32)
y_test=torch.tensor(y_test)
y_test=y_test[-ntest:,...].to(torch.float32)
print("sing fidelity:",fid)
print("training samples:",x_train.shape)
print("testing samples:",x_test.shape)
model = FNO3d(modes1=12, modes2=12,modes3=2, width=32).cuda()
x_normalizer,y_normalizer,folder=model.train_model(x_train,y_train,batch_size=bsize,epochs=ep,work_dir=f'single_{fid}_{ntrain}_{seed}')
pt_path=os.path.join(folder,'model.pt')
torch.save([x_normalizer,model,y_normalizer], pt_path)
model.test_model(x_test,y_test,x_normalizer,y_normalizer)







