import multiprocessing
import numpy as np
import time
import random

import os
# def fun(fid,train_num,seed,epoch,bsize):
#     os.system(f'python part_single_train.py train_num={train_num} fid={fid} seed={seed} epoch={epoch} batch_size={bsize}')
def unfun(low_num,high_num,seed,epoch,bsize):
    os.system(f'python new_union_train.py low_num={low_num} high_num={high_num} seed={seed} epoch={epoch} batch_size={bsize}')

if __name__ == '__main__':

    seed_pol = [1]

    for seedx in seed_pol:
        ##########################################
        print("cost 2400")
        print('union 1:2')
        p = multiprocessing.Process(target=unfun(low_num=800, high_num=400, seed=seedx, epoch=400, bsize=10))
        p.start()
        print('union 1:3')
        p = multiprocessing.Process(target=unfun(low_num=990, high_num=342, seed=seedx, epoch=400, bsize=10))
        p.start()
        print('union 1:4')
        p = multiprocessing.Process(target=unfun(low_num=990, high_num=300, seed=seedx, epoch=400, bsize=10))
        p.start()

