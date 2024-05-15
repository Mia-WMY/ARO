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
x_train = f'../core8_steady/{fid}_x_train.npy'
y_train=f'../core8_steady/{fid}_y_train.npy'
x_test = f'../core8_steady/high_x_test.npy'
y_test=f'../core8_steady/high_y_test.npy'

################################################################
# load data and data normalization
################################################################
import os
from fourier_mcdropout.utilities3 import *
x_train=np.load(x_train)
x_train=torch.tensor(x_train)
x_train=x_train[:ntrain,...].to(torch.float32)
y_train = np.load(y_train)
y_train=torch.tensor(y_train)
y_train=y_train[:ntrain,...].to(torch.float32)
x_test = np.load(x_test)
x_test=torch.tensor(x_test)
x_test=x_test[:ntest,...].to(torch.float32)
y_test = np.load(y_test)
y_test=torch.tensor(y_test)
y_test=y_test[:ntest,...].to(torch.float32)
print("sing fidelity:",fid)
print("training samples:",x_train.shape)
print("testing samples:",x_test.shape)
model = FNO3d(modes1=12, modes2=12,modes3=2, width=32).cuda()
x_normalizer,y_normalizer,folder=model.train_model(x_train,y_train,batch_size=bsize,epochs=ep,work_dir=f'single_{fid}_{ntrain}_{seed}')
pt_path=os.path.join(folder,'model.pt')
torch.save([x_normalizer,model,y_normalizer], pt_path)
model.test_model(x_test,y_test,x_normalizer,y_normalizer)







