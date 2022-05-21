import sys, os
import torch
import torch.nn as nn
import argparse

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LDP_CODESIGN_DIR = os.environ['LDP_CODESIGN_DIR'] 
sys.path.append(LDP_CODESIGN_DIR)
sys.path.append(LDP_CODESIGN_DIR + '/utils/')
sys.path.append(LDP_CODESIGN_DIR + '/codesign/')

SCRATCH_DIR = LDP_CODESIGN_DIR + '/scratch/'

from utils import *

from textfile_utils import *
from plotting_utils import *

from collections import OrderedDict

from numpy.linalg import eig


def regressor_train(regressor, data_dict, train_options):
    data_dim = data_dict['data_dim']

    x = torch.tensor(data_dict['train_dataset'], dtype=torch.float32).to(device)
    y = torch.tensor(data_dict['train_outputs'], dtype=torch.float32).to(device)

    num_samples = x.shape[0]

    regressor_optimizer = torch.optim.Adam(
        regressor.parameters(), lr=train_options["learning_rate"], amsgrad=True)
   
    loss_fn = torch.nn.MSELoss().to(device)

    train_losses = []
    for i in range(train_options["num_epochs"]):

        y_hat = regressor(x)

        train_loss = loss_fn(y_hat, y)

        regressor_optimizer.zero_grad()
        train_loss.backward()
        regressor_optimizer.step()

        train_losses.append(train_loss.item())
        
        if (i + 1) % train_options["output_freq"] == 0:
            print("Epoch: {} train_loss: {}\n".format(i+1, train_losses[-1]))

    return train_losses, regressor


def regressor_test(regressor, data_dict):
    data_dim = data_dict['data_dim']

    x = torch.tensor(data_dict['test_dataset'], dtype=torch.float32).to(device)
    y = torch.tensor(data_dict['test_outputs'], dtype=torch.float32).to(device)

    num_samples = x.shape[0]

    loss_fn = torch.nn.MSELoss().to(device)

    y_hat = regressor(x)
    test_loss = loss_fn(y_hat, y)

    print("test_loss: {}\n".format(test_loss.item()))

    return test_loss


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--regressor_name', type=str)
    parser.add_argument('--num_epochs', type=int, default=500)
    args = parser.parse_args()

    model_name = args.model_name
    regressor_name = args.regressor_name
    num_epochs = args.num_epochs

    BASE_DIR = SCRATCH_DIR + model_name
    DATA_DIR = BASE_DIR + '/dataset/'
    data_dict = load_pkl(DATA_DIR + '/data.pkl')

    train_options = { 
                      "num_epochs": num_epochs,
                      "learning_rate": 1e-3,
                      "output_freq": 10,
                      "save_model": True
                    }

    TEST_DATA_DIR = BASE_DIR + '/regressor/'
    remove_and_create_dir(TEST_DATA_DIR)

    data_dim = data_dict['data_dim']
    regressor_paras = {"input_dim": data_dim}
    regressor = init_regressor(regressor_name, regressor_paras)

    train_losses, regressor = regressor_train(
        regressor, data_dict, train_options)

    test_loss = regressor_test(regressor, data_dict)

    result_dict = OrderedDict()
    result_dict['regressor_name'] = regressor_name
    result_dict['regressor_paras'] = regressor_paras
    result_dict['regressor_state_dict'] = regressor.state_dict()
    result_dict['train_losses'] = train_losses
    result_dict['test_loss'] = test_loss

    write_pkl(fname = TEST_DATA_DIR + '/regressor.pkl', 
              input_dict = result_dict)
