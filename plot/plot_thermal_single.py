'''
保存瞬态模型测试样本的瞬态预测,并对其动态预测绘图
可绘制样本中layer1~3的四个不同时刻预测热力图

内附一个检测数据集分布的代码
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import logging
import numpy as np
import torch
import os
from fourier_mcdropout.utilities3 import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import mean_squared_error,r2_score
torch.manual_seed(0)
np.random.seed(0)
def predict_union(load_models,x_test,s1,s2,batch_size):
    aro=load_models[0]
    x_normalizer_low = load_models[1]
    model_low = load_models[2]
    y_normalizer_low = load_models[3]
    x_normalizer_high = load_models[4]
    model_res = load_models[5]
    y_normalizer_high = load_models[6]
    x_test_low = x_normalizer_low.forward(x_test)
    x_test_high = x_normalizer_high.forward(x_test)
    x_low, x_high= x_test_low.cuda(), x_test_high.cuda()
    out_low = model_low(x_low).reshape(batch_size, s1, s2,2)
    out_low = y_normalizer_low.inverse(out_low)
    out_high = model_res(x_high).reshape(batch_size, s1, s2,2)
    out_high = y_normalizer_high.inverse(out_high)
    out=out_low+out_high
    return out
def cal_rmse(y_true,y_pred):
    mse=np.mean((y_true-y_pred)**2)
    rmse=np.sqrt(mse)
    return rmse.item()
default_dtype = np.float64
import matplotlib.pyplot as plt

# 设置全局字体大小
plt.rcParams.update({'font.size': 12})
if __name__ == '__main__':
    # loading data
    x_test = '../aro_data/ev6_single/high_x_test.npy'
    y_test = '../aro_data/ev6_single/high_y_test.npy'
    x_test = np.load(x_test)
    y_test = np.load(y_test)
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)
    union_model = torch.load('../experiments_on_ev6_single/union_low600_high150_1/model.pt')
    #### settings
    s1 = 87
    s2 = 87
    batch_size = 20
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                                   batch_size=batch_size, shuffle=False, drop_last=False)


    origin_y=[]
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        union_out=predict_union(union_model,x,s1,s2,x.shape[0])
        origin_y=y




colors = [(0, 'blue'), (1 / 6, 'cyan'), (2 / 6, 'green'), (3 / 6, 'yellow'), (4 / 6, 'orange'), (5 / 6, 'red'),
          (1, 'darkred')]
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
#
colors = [(0, 'white'), (1 / 6, '#87CFF5'), (2 / 6, '#B8DDCC'), (3 / 6, '#FFEFBE'), (4 / 6, '#F8D5B7'), (5 / 6, '#EBA9AB'),
          (1, '#EBA9AB')]
cmapx = LinearSegmentedColormap.from_list('custom_cmap', colors)
index=[2,5,9,14]
fig, axes = plt.subplots(4, 3, figsize=(7, 6.5))
layer=1
for i in range(4):
    matrix_union =union_out[index[i], :, :].cpu().detach().numpy()
    origin_high=origin_y[index[i],:,:].cpu().detach().numpy()

    vmin0 = np.min(origin_high)
    vmax0 = np.max(origin_high)
    # axes[ 0].imshow(origin_low, cmap=cmap, interpolation='nearest', vmin=vmin0, vmax=vmax0)

    im1=axes[i,0].imshow( np.rot90(origin_high[...,layer]), cmap=cmap, interpolation='nearest', vmin=vmin0, vmax=vmax0)
    fig.colorbar(im1, ax=axes[i,0], location='left')
    axes[i,1].imshow( np.rot90(matrix_union[...,layer]), cmap=cmap, interpolation='nearest', vmin=vmin0, vmax=vmax0)
    # axes[0,2].imshow(origin_high[...,2], cmap=cmap, interpolation='nearest', vmin=vmin0, vmax=vmax0)
    abs=np.abs(origin_high[...,layer]-matrix_union[...,layer])

    im2=axes[i,2].imshow( np.rot90(abs), cmap=cmapx, interpolation='nearest', vmin=0.1, vmax=1)
    fig.colorbar(im2, ax=axes[i, 2], location='right')
    # axes[1,1].imshow( np.rot90(matrix_union[...,1]), cmap=cmap, interpolation='nearest', vmin=vmin0, vmax=vmax0)
    # axes[1,2].imshow(matrix_union[...,2], cmap=cmap, interpolation='nearest', vmin=vmin0, vmax=vmax0)
    axes[i,0].axis('off')
    axes[i,1].axis('off')
    axes[i,2].axis('off')
#
# axes[1,0].axis('off')
# axes[1,1].axis('off')
axes[0,0].set_title(f"Ground Truth", fontsize=14)
axes[0,1].set_title(f"Predict", fontsize=14)
axes[0,2].set_title(f"Error", fontsize=14)
# axes[0, 0].text(-10,50, "Layer1", fontsize=16, va='center', ha='center', rotation=90)
# axes[1, 0].text(-10,50, "Layer2", fontsize=16, va='center', ha='center', rotation=90)
plt.subplots_adjust(wspace=0.01, )
# rmse= round(cal_rmse(origin_high[...,0], matrix_union[...,0]),2)
# r2=round(r2_score(origin_high[...,0],matrix_union[...,0]),2)
# print("layer 1",rmse,r2)
# rmse = round(cal_rmse(origin_high[..., 1], matrix_union[..., 1]), 1)
# r2 = round(r2_score(origin_high[..., 1], matrix_union[..., 1]), 1)
# print("layer 2",rmse,r2)
#
# plt.xticks([])
# plt.yticks([])
# cbar_ax = fig.add_axes([0.12, 0.05, 0.78, 0.03])
# fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

# plt.tight_layout()
save_pth = f'compare_4case.pdf'
plt.savefig(save_pth)

plt.show()
