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

    file = LDP_CODESIGN_DIR + '/data/cancer/wdbc.csv'
    df = pd.read_csv(file)

    raw_data_np = df.to_numpy()
    raw_data_np = raw_data_np[:, 1:]

    num_samples = raw_data_np.shape[0]
    data_dim = raw_data_np.shape[1]-1

    minmaxscaler = MinMaxScaler()
    raw_data_np = minmaxscaler.fit_transform(raw_data_np)
    # raw_data_np[:, :data_dim] = minmaxscaler.fit_transform(raw_data_np[:, :data_dim])

    print(raw_data_np)
    print(raw_data_np.shape)

    num_train_samples = int(num_samples * 0.7)
    num_test_samples = num_samples - num_train_samples

    train_dataset = raw_data_np[:num_train_samples, 1:]
    train_tags = raw_data_np[:num_train_samples, 0]
    print(train_dataset.shape)

    test_dataset = raw_data_np[num_train_samples:, 1:]
    test_tags = raw_data_np[num_train_samples:, 0]
    print(test_dataset.shape)

    num_clusters = 2

    RESULTS_DIR = SCRATCH_DIR + '/dataset/'
    remove_and_create_dir(RESULTS_DIR)

    data_dict = OrderedDict()

    data_dict['train_dataset'] = train_dataset
    data_dict['test_dataset'] = test_dataset

    data_dict['train_tags'] = train_tags
    data_dict['test_tags'] = test_tags
    data_dict['num_clusters'] = num_clusters

    data_dict['data_dim'] = data_dim

    # data_dict['task'] = { 
    #                         "task_name": "matrix_multiplication",
    #                         "K": K
    #                     }

    data_dict['task'] = { 
                            "task_name": "classification",
                        }

    write_pkl(fname = RESULTS_DIR + '/data.pkl', input_dict = data_dict)
