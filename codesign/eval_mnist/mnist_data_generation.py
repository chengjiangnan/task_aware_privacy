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

from keras.datasets import mnist


np.random.seed(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    
    model_name = args.model_name

    SCRATCH_DIR = LDP_CODESIGN_DIR + '/scratch/' + model_name

    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    num_train_samples = train_X.shape[0]
    num_test_samples = test_X.shape[0]
    data_dim = train_X.shape[1] * train_X.shape[2]

    train_dataset = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2])
    print(train_dataset.shape)

    test_dataset = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2])
    print(test_dataset.shape)

    train_dataset = train_dataset / 255
    test_dataset = test_dataset / 255

    train_tags = train_y
    test_tags = test_y

    # To save training time, We only used 10% of the MNIST dataset
    num_train_samples = 6000
    num_test_samples = 1000
    print("Number of data samples used in experiment:")
    print("Training: {}, Testing: {}"
          .format(num_train_samples, num_test_samples))

    train_dataset = train_dataset[:num_train_samples, :]
    test_dataset = test_dataset[:num_test_samples, :]

    train_tags = train_tags[:num_train_samples]
    test_tags = test_tags[:num_test_samples]

    num_clusters = 10

    RESULTS_DIR = SCRATCH_DIR + '/dataset/'
    remove_and_create_dir(RESULTS_DIR)

    data_dict = OrderedDict()

    data_dict['train_dataset'] = train_dataset
    data_dict['test_dataset'] = test_dataset

    data_dict['train_tags'] = train_tags
    data_dict['test_tags'] = test_tags
    data_dict['num_clusters'] = num_clusters

    data_dict['data_dim'] = data_dim

    data_dict['task'] = { 
                            "task_name": "classification",
                        }

    write_pkl(fname = RESULTS_DIR + '/data.pkl', input_dict = data_dict)
