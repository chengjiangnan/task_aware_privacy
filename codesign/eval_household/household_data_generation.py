import numpy as np
import pandas as pd
import os, sys
from collections import OrderedDict
import torch
from torch.utils.data import TensorDataset, DataLoader

import math
from scipy import signal

import argparse

LDP_CODESIGN_DIR = os.environ['LDP_CODESIGN_DIR'] 
sys.path.append(LDP_CODESIGN_DIR)
sys.path.append(LDP_CODESIGN_DIR + '/utils/')
sys.path.append(LDP_CODESIGN_DIR + '/codesign/')

from textfile_utils import *
from plotting_utils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm

from numpy.linalg import eig, eigh, inv


np.random.seed(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    
    model_name = args.model_name

    SCRATCH_DIR = LDP_CODESIGN_DIR + '/scratch/' + model_name

    file = LDP_CODESIGN_DIR + '/data/household/household_power_consumption_aggre_clean.csv'
    df = pd.read_csv(file)

    power_list = df['Power'].values

    T = 24
    num_values = len(power_list)
    assert(num_values % T == 0)

    minmaxscaler = MinMaxScaler()
    power_list = minmaxscaler.fit_transform(power_list.reshape(num_values, 1))

    num_samples = int(num_values / T)
    power_matrix = power_list.reshape(num_samples, T)

    print(power_matrix)

    print(np.mean(power_matrix, axis=0))
    print(np.var(power_matrix, axis=0))

    data_dim = T

    num_train_samples = int(num_samples * 0.7)
    num_test_samples = num_samples - num_train_samples

    train_dataset = power_matrix[:num_train_samples, :]
    test_dataset = power_matrix[num_train_samples:, :]

    K = np.identity(T)
    for i in range(T):
        K[i, i] = 1 if (i < 8 or i >= 20) else 2

    RESULTS_DIR = SCRATCH_DIR + '/dataset/'
    remove_and_create_dir(RESULTS_DIR)

    data_dict = OrderedDict()

    data_dict['train_dataset'] = train_dataset
    data_dict['test_dataset'] = test_dataset

    data_dict['data_dim'] = data_dim

    data_dict['task'] = { 
                            "task_name": "matrix_multiplication",
                            "K": K
                        }

    # data_dict['task'] = { 
    #                         "task_name": "classification",
    #                       }

    write_pkl(fname = RESULTS_DIR + '/data.pkl', input_dict = data_dict)
