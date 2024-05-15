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
    out_low = model_low(x_low).reshape(batch_size, s1, s2,2,9)
    out_low = y_normalizer_low.inverse(out_low)
    out_high = model_res(x_high).reshape(batch_size, s1, s2,2,9)
    out_high = y_normalizer_high.inverse(out_high)
    out=out_low+out_high
    return out
def cal_rmse(y_true,y_pred):
    mse=np.mean((y_true-y_pred)**2)
    rmse=np.sqrt(mse)
    return rmse.item()
default_dtype = np.float64

if __name__ == '__main__':
    # loading data
    x_test = '../aro_data/ev6_single_trans/high_x_test.npy'
    y_test = '../aro_data/ev6_single_trans/high_y_test.npy'
    x_test = np.load(x_test)
    y_test = np.load(y_test)
    x_test = torch.tensor(x_test)
    x_test=x_test[:100,...]
    y_test = torch.tensor(y_test)
    y_test=y_test[:100,...]
    union_model = torch.load('../experiments_on_ev6_single_trans/union_low500_high125/model.pt')
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



import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
colors = [(0, 'blue'), (1 / 6, 'cyan'), (2 / 6, 'green'), (3 / 6, 'yellow'), (4 / 6, 'orange'), (5 / 6, 'red'),
          (1, 'darkred')]
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
for i in range(20):
    # fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    matrix_union =union_out[i, ...].cpu().detach().numpy()
    origin_high=origin_y[i,...].cpu().detach().numpy()
    # fig = plt.figure()
    layer_num=matrix_union.shape[-2]
    fig, ax = plt.subplots(2, 5, figsize=(8, 3), frameon=False)
    ax[0,0].axis("off")
    ax[0, 1].axis("off")
    ax[0, 2].axis("off")
    ax[0, 3].axis("off")
    ax[0, 4].axis("off")
    ax[1, 0].axis("off")
    ax[1, 1].axis("off")
    ax[1, 2].axis("off")
    ax[1, 3].axis("off")
    ax[1, 4].axis("off")
    axes=ax
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    axes[0, 0].set_title(f"t=0", fontsize=8)
    axes[0, 1].set_title(f"t=2", fontsize=8)
    axes[0, 2].set_title(f"t=4", fontsize=8)
    axes[0, 3].set_title(f"t=6", fontsize=8)
    axes[0, 4].set_title(f"t=8", fontsize=8)

    axes[0, 0].text(-0.2, 0.4, "Ground Truth", fontsize=8, va='center', ha='center', rotation=90)
    print(axes[0,0].get_xlim(),axes[1,0].get_ylim())
    axes[1, 0].text(-0.2, 0.4, "Predict", fontsize=8, va='center', ha='center', rotation=90)
    t=0

    colors = [(0, 'blue'), (1 / 6, 'cyan'), (2 / 6, 'green'), (3 / 6, 'yellow'), (4 / 6, 'orange'), (5 / 6, 'red'),
              (1, 'darkred')]
    cmaps = LinearSegmentedColormap.from_list('custom_cmap', colors)
    vmin0 = np.min(origin_high[...,0,:])
    vmax0 = np.max(origin_high[...,0,:])
    truth=origin_high[...,t]
    pred=matrix_union[...,t]

    # 获取热量模型矩阵的形状
    x_size, y_size, z_size = truth.shape
    # 创建x、y、z坐标数组
    x = np.arange(x_size)
    y = np.arange(y_size)
    z = np.arange(z_size)

    # 创建网格
    X, Y, Z = np.meshgrid(x, y, z)

    # 将热量模型数据展平，以便进行绘图
    truth0 =origin_high[...,0].ravel()
    pred0 =matrix_union[..., 0].ravel()
    # 绘制3D立体图
    # print(origin_high[...,0].shape)


    ax = fig.add_subplot(2, 5, 1, projection='3d',frame_on=False)
    ax.scatter(X, Y, Z, c=truth0, cmap=cmaps,vmax=vmax0,vmin=vmin0)
    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()


    ax.axis("off")
    ax = fig.add_subplot(2, 5, 6, projection='3d',frame_on=False)
    ax.scatter(X, Y, Z, c=pred0, cmap=cmaps,vmax=vmax0,vmin=vmin0)
    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    # ax.gca().spines['top'].set_visible(False)
    # ax.gca().spines['right'].set_visible(False)
    # ax.gca().spines['bottom'].set_visible(False)
    # ax.gca().spines['left'].set_visible(False)
    ###
    truth0 =origin_high[...,2].ravel()
    pred0 = matrix_union[..., 2].ravel()
    # 绘制3D立体图

    ax.axis("off")
    ax = fig.add_subplot(2, 5, 2, projection='3d',frame_on=False)
    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    # ax.gca().spines['top'].set_visible(False)
    # ax.gca().spines['right'].set_visible(False)
    # ax.gca().spines['bottom'].set_visible(False)
    # ax.gca().spines['left'].set_visible(False)
    ax.scatter(X, Y, Z, c=truth0, cmap=cmaps,vmax=vmax0,vmin=vmin0)

    ax = [1, 1]
    ax = fig.add_subplot(2, 5, 7, projection='3d',frame_on=False)
    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.scatter(X, Y, Z, c=pred0, cmap=cmaps,vmax=vmax0,vmin=vmin0)
    ax.set_axis_off()
    ###
    truth0 =origin_high[...,4].ravel()
    pred0 = matrix_union[...,4 ].ravel()
    # 绘制3D立体图
    # ax[0,2].scatter(X, Y, Z, c=truth0, cmap='hot')
    # ax[1, 2].scatter(X, Y, Z, c=pred0, cmap='hot')

    ax = [0, 2]
    ax = fig.add_subplot(2, 5, 3, projection='3d',frame_on=False)
    ax.scatter(X, Y, Z, c=truth0, cmap=cmaps,vmax=vmax0,vmin=vmin0)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis("off")
    ax.set_axis_off()
    # ax.gca().spines['top'].set_visible(False)
    # ax.gca().spines['right'].set_visible(False)
    # ax.gca().spines['bottom'].set_visible(False)
    # ax.gca().spines['left'].set_visible(False)
    ax = [1, 2]
    ax = fig.add_subplot(2, 5, 8, projection='3d')
    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    # ax.gca().spines['top'].set_visible(False)
    # ax.gca().spines['right'].set_visible(False)
    # ax.gca().spines['bottom'].set_visible(False)
    # ax.gca().spines['left'].set_visible(False)
    ax.scatter(X, Y, Z, c=pred0, cmap=cmaps,vmax=vmax0,vmin=vmin0)
    truth0 =origin_high[...,6].ravel()
    pred0 = matrix_union[..., 6].ravel()

    ax = [0, 3]
    ax = fig.add_subplot(2, 5, 4, projection='3d',frame_on=False)
    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    # ax.gca().spines['top'].set_visible(False)
    # ax.gca().spines['right'].set_visible(False)
    # ax.gca().spines['bottom'].set_visible(False)
    # ax.gca().spines['left'].set_visible(False)
    ax.scatter(X, Y, Z, c=truth0, cmap=cmaps,vmax=vmax0,vmin=vmin0)

    ax = [1, 3]
    ax = fig.add_subplot(2, 5, 9, projection='3d',frame_on=False)
    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.scatter(X, Y, Z, c=pred0, cmap=cmaps,vmax=vmax0,vmin=vmin0)
    truth0 = origin_high[..., 8].ravel()
    pred0 = matrix_union[..., 8].ravel()

    ax = [0, 3]
    ax = fig.add_subplot(2, 5, 5, projection='3d', frame_on=False)
    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    # ax.gca().spines['top'].set_visible(False)
    # ax.gca().spines['right'].set_visible(False)
    # ax.gca().spines['bottom'].set_visible(False)
    # ax.gca().spines['left'].set_visible(False)
    ax.scatter(X, Y, Z, c=truth0, cmap=cmaps, vmax=vmax0, vmin=vmin0)

    ax = [1, 3]
    ax = fig.add_subplot(2, 5, 10, projection='3d', frame_on=False)
    ax.grid(False)
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.scatter(X, Y, Z, c=pred0, cmap=cmaps, vmax=vmax0, vmin=vmin0)
    # ax.gca().spines['bottom'].set_visible(False)
    plt.savefig(f"trans_{i}.pdf")
    plt.axis('off')
    # ax.gca().spines['bottom'].set_visible(False)
    plt.subplots_adjust(wspace=0.001, hspace=0.00001)
    plt.show()
