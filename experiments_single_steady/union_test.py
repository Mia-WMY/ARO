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
x_train_low = '../ev6_single/med_x_train.npy'
x_test_low= '../ev6_single/med_x_test.npy'
y_train_low = '../ev6_single/med_y_train.npy'
y_test_low= '../ev6_single/med_y_test.npy'
###############################################
x_train_high = '../ev6_single/high_x_train.npy'
x_test_high= '../ev6_single/high_x_test.npy'
y_train_high = '../ev6_single/high_y_train.npy'
y_test_high= '../ev6_single/high_y_test.npy'

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
ntrain_high=1400
ntrain_low=350
seed=1
setup_seed(seed)
from fourier_mcdropout.utilities3 import *

ntest=500
x_train_low=np.load(x_train_low)
x_train_low=torch.tensor(x_train_low)
x_train_low=x_train_low[:ntrain_low,...]

y_train_low =np.load(y_train_low)
y_train_low=torch.tensor(y_train_low)
y_train_low=y_train_low[:ntrain_low,...]

x_test_low =np.load(x_test_low)
x_test_low=torch.tensor(x_test_low).to('cuda')
x_test_low=x_test_low[:ntest,...].to(torch.float32)
y_test_low = np.load(y_test_low)
y_test_low=torch.tensor(y_test_low).to('cuda')
y_test_low=y_test_low[:ntest,...].to(torch.float32)
x_train_low=x_train_low.to('cuda')
x_test_low=x_test_low.to('cuda')

### high
x_train_high=np.load(x_train_high)
x_train_high=torch.tensor(x_train_high)
x_train_high=x_train_high[:ntrain_high,...].to(torch.float32)

y_train_high = np.load(y_train_high)
y_train_high=torch.tensor(y_train_high)
y_train_high=y_train_high[:ntrain_high,...].to(torch.float32)

x_test_high =np.load(x_test_high)
x_test_high=torch.tensor(x_test_high).to('cuda')
x_test_high=x_test_high[:ntest,...].to(torch.float32)
y_test_high =np.load(y_test_high)
y_test_high=torch.tensor(y_test_high).to('cuda')
y_test_high=y_test_high[:ntest,...].to(torch.float32)
x_train_high=x_train_high.to('cuda')
x_test_high=x_test_high.to('cuda')


models=torch.load("union_low667_high333_1/model.pt")
aro=models[0]
x_normalizer_low=models[1]
model_low=models[2]
y_normalizer_low=models[3]
x_normalizer_res=models[4]
model_res=models[5]
y_normalizer_res=models[6]
t1=default_timer()
aro.valid_model(x_test_high,y_test_high,model_low,model_res,x_normalizer_low,y_normalizer_low,x_normalizer_res,y_normalizer_res)
t2=default_timer()
print(t2-t1)
