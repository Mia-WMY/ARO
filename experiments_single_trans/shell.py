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
        ##########################################
        print("seed 1")
        print("---------------------------------------------------------------------")
        # print("cost 400")

        # print('union 1:2')
        # p = multiprocessing.Process(target=unfun(low_num=133, high_num=67, seed=seedx, epoch=800, bsize=10))
        # p.start()
        # print("single low")
        # p = multiprocessing.Process(target=fun(fid='mid', train_num=400, seed=seedx, epoch=800, bsize=10))
        # p.start()
     #    print("single high")
     #    p = multiprocessing.Process(target=fun(fid='high', train_num=100, seed=seedx, epoch=800, bsize=10))
     #    p.start()
     #    print('union 1:3')
     #    p = multiprocessing.Process(target=unfun(low_num=171, high_num=57, seed=seedx, epoch=800, bsize=10))
     #    p.start()
     #    print('union 1:4')
     #    p = multiprocessing.Process(target=unfun(low_num=200, high_num=50, seed=seedx, epoch=800, bsize=10))
     #    p.start()
     # #####################3
     #    print("cost 800")
     #    print("single low")
     #    p = multiprocessing.Process(target=fun(fid='mid', train_num=800, seed=seedx, epoch=800, bsize=10))
     #    p.start()
     #    print("single high")
     #    p = multiprocessing.Process(target=fun(fid='high', train_num=200, seed=seedx, epoch=800, bsize=10))
     #    p.start()
     #    print('union 1:2')
     #    p = multiprocessing.Process(target=unfun(low_num=267, high_num=133, seed=seedx, epoch=800, bsize=10))
     #    p.start()
     #    print('union 1:3')
     #    p = multiprocessing.Process(target=unfun(low_num=342, high_num=114, seed=seedx, epoch=800, bsize=10))
     #    p.start()
        print('union 1:4')
        p = multiprocessing.Process(target=unfun(low_num=400, high_num=100, seed=seedx, epoch=800, bsize=10))
        p.start()
   #####################3
   #      print("cost 1200")
   #      print("single low")
   #      p = multiprocessing.Process(target=fun(fid='mid', train_num=1200, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print("single high")
   #      p = multiprocessing.Process(target=fun(fid='high', train_num=300, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:2')
   #      p = multiprocessing.Process(target=unfun(low_num=400, high_num=200, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:3')
   #      p = multiprocessing.Process(target=unfun(low_num=514, high_num=171, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:4')
   #      p = multiprocessing.Process(target=unfun(low_num=600, high_num=150, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   # #####################3
   #      print("cost 1600")
   #      print("single low")
   #      p = multiprocessing.Process(target=fun(fid='mid', train_num=1600, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print("single high")
   #      p = multiprocessing.Process(target=fun(fid='high', train_num=400, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:2')
   #      p = multiprocessing.Process(target=unfun(low_num=533, high_num=267, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:3')
   #      p = multiprocessing.Process(target=unfun(low_num=685, high_num=229, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:4')
   #      p = multiprocessing.Process(target=unfun(low_num=800, high_num=200, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print("cost 2000")
   #      print("single low")
   #      p = multiprocessing.Process(target=fun(fid='mid', train_num=2000, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print("single high")
   #      p = multiprocessing.Process(target=fun(fid='high', train_num=500, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:2')
   #      p = multiprocessing.Process(target=unfun(low_num=667, high_num=333, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:3')
   #      p = multiprocessing.Process(target=unfun(low_num=857, high_num=285, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:4')
   #      p = multiprocessing.Process(target=unfun(low_num=1000, high_num=250, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print("cost 2400")
   #      print("single low")
   #      p = multiprocessing.Process(target=fun(fid='mid', train_num=2400, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print("single high")
   #      p = multiprocessing.Process(target=fun(fid='high', train_num=600, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:2')
   #      p = multiprocessing.Process(target=unfun(low_num=800, high_num=400, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:3')
   #      p = multiprocessing.Process(target=unfun(low_num=1028, high_num=343, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:4')
   #      p = multiprocessing.Process(target=unfun(low_num=1200, high_num=300, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print("cost 2800")
   #      print("single low")
   #      p = multiprocessing.Process(target=fun(fid='mid', train_num=2800, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print("single high")
   #      p = multiprocessing.Process(target=fun(fid='high', train_num=700, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:2')
   #      p = multiprocessing.Process(target=unfun(low_num=933, high_num=467, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:3')
   #      p = multiprocessing.Process(target=unfun(low_num=1200, high_num=400, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:4')
   #      p = multiprocessing.Process(target=unfun(low_num=1400, high_num=350, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print("cost 3200")
   #      print("single low")
   #      p = multiprocessing.Process(target=fun(fid='mid', train_num=3200, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print("single high")
   #      p = multiprocessing.Process(target=fun(fid='high', train_num=800, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:2')
   #      p = multiprocessing.Process(target=unfun(low_num=1067, high_num=533, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:3')
   #      p = multiprocessing.Process(target=unfun(low_num=1371, high_num=457, seed=seedx, epoch=800, bsize=10))
   #      p.start()
   #      print('union 1:4')
   #      p = multiprocessing.Process(target=unfun(low_num=1600, high_num=400, seed=seedx, epoch=800, bsize=10))
   #      p.start()