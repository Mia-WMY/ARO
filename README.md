# ARO: Autoregressive Operator Learning for Transferable and Multi-fidelity 3D-IC Thermal Analysis with Active Learning

## Introduction

🤔🤔🤔

This is the code implementation of the paper **“ARO: Autoregressive Operator Learning for Transferable and Multi-fidelity 3D-IC Thermal Analysis with Active Learning”**.

The structure of the project is as follows：

-- active_learning_for_steady (the implementation of active learing for steady experiments)

-- experiments_3d_steady (experiments on *3d_steady*)

-- experiments_3d_trans (experiments on *3d_trans*)

-- experiments_core8_steady (experiments on *core8_steady*)

-- experiments_core8_trans (experiments on *core8_trans*)

-- experiments_single_steady (experiments on *single_steady*)

-- experiments_single_trans (experiments on *single_trans*)

-- fourier_mcdropout (the implementation of Fourier Neural Operator)

-- plot (the source code of plotting)

*3d_steady*, *3d_trans*, *core8_steady*, *core8_trans*, *single_steady* and *single_trans* are six datasets on three 3D-ICs used in this work.


## Requirements

🥸🥸🥸

This project bases on **Python 3.10.8**.

## Usage

😇😇😇

The structures of documents started by **experiments** are similar, taking **experiments_3d_steady**  as an example :

-- single_train.py (the training of single-fidelity)

-- union_train.py (the training of multi-fidelity)

-- shell.py (the script for both single-fidelity training and multi-fildeity training)
