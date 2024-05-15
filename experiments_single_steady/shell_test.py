import multiprocessing
import numpy as np
import time
import random

import os
def fun(fid,train_num,seed,epoch,bsize):
    os.system(f'python single_train.py train_num={train_num} fid={fid} seed={seed} epoch={epoch} batch_size={bsize}')
def unfun(low_num,high_num,seed,epoch,bsize):
    os.system(f'python union_train.py low_num={low_num} high_num={high_num} seed={seed} epoch={epoch} batch_size={bsize}')

if __name__ == '__main__':
    seed_pol=[1,2,3]

    for seedx in seed_pol:
        ##########################################
        print("seed 1")
        print("---------------------------------------------------------------------")
        print("cost 400")
        print("single low")
        p = multiprocessing.Process(target=fun(fid='med', train_num=400, seed=seedx, epoch=8, bsize=20))
        p.start()
        print("single high")
        p = multiprocessing.Process(target=fun(fid='high',train_num=100, seed=seedx, epoch=8, bsize=20))
        p.start()
        print('union 1:2')
        p = multiprocessing.Process(target=unfun(low_num=133, high_num=67, seed=seedx, epoch=8, bsize=20))
        p.start()
        print('union 1:3')
        p = multiprocessing.Process(target=unfun(low_num=171, high_num=57, seed=seedx, epoch=8, bsize=20))
        p.start()
        print('union 1:4')
        p = multiprocessing.Process(target=unfun(low_num=200, high_num=50, seed=seedx, epoch=8, bsize=20))
        p.start()
