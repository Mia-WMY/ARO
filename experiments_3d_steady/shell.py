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
    seed_pol=[1]

    for seedx in seed_pol:
        print("cost 4000")
        print("single low")
        p = multiprocessing.Process(target=fun(fid='mid', train_num=4000, seed=seedx, epoch=800, bsize=20))
        p.start()
        print("single high")
        p = multiprocessing.Process(target=fun(fid='high',train_num=1000, seed=seedx, epoch=800, bsize=20))
        p.start()
        print('union 1:2')
        p = multiprocessing.Process(target=unfun(low_num=1333, high_num=666, seed=seedx, epoch=800, bsize=20))
        p.start()
        print('union 1:3')
        p = multiprocessing.Process(target=unfun(low_num=1714, high_num=571, seed=seedx, epoch=800, bsize=20))
        p.start()
        print('union 1:4')
        p = multiprocessing.Process(target=unfun(low_num=2000, high_num=500, seed=seedx, epoch=800, bsize=20))
        p.start()