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


def classifier_train(classifier, data_dict, train_options):
    data_dim = data_dict['data_dim']

    x = torch.tensor(data_dict['train_dataset'], dtype=torch.float32).to(device)
    tags = torch.tensor(data_dict['train_tags'], dtype=torch.long).to(device)

    # use batch
    training_set = Dataset(x, tags)
    training_generator = torch.utils.data.DataLoader(
        training_set, batch_size=64, shuffle=True, num_workers=4)

    num_samples = x.shape[0]

    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(), lr=train_options["learning_rate"], amsgrad=True)
   
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    train_losses = []
    for i in range(train_options["num_epochs"]):
        j = 0
        for batch_x, batch_tags in training_generator:
            batch_tags_hat = classifier(batch_x)

            batch_train_loss = loss_fn(batch_tags_hat, batch_tags)

            classifier_optimizer.zero_grad()
            batch_train_loss.backward()
            classifier_optimizer.step()
            
            j = j + 1

            if (j + 1) % train_options["output_freq"] == 0:
                print("Epoch: {}, Batch: {}, Batch_train_loss: {}\n"
                      .format(i, j, batch_train_loss.item()))

        if (i + 1) % 10 == 0:
            tags_hat = classifier(x)
            train_loss = loss_fn(tags_hat, tags)
            train_losses.append(train_loss.item())

            print("Epoch: {}, Train_loss: {}\n".format(i, train_losses[-1]))

    return train_losses, classifier


def classifier_test(classifier, data_dict):
    data_dim = data_dict['data_dim']

    x = torch.tensor(data_dict['test_dataset'], dtype=torch.float32).to(device)
    tags = torch.tensor(data_dict['test_tags'], dtype=torch.long).to(device)

    num_samples = x.shape[0]

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    tags_hat = classifier(x)
    test_loss = loss_fn(tags_hat, tags)

    print("test_loss: {}\n".format(test_loss.item()))

    return test_loss


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--classifier_name', type=str)
    parser.add_argument('--num_epochs', type=int, default=500)
    args = parser.parse_args()

    model_name = args.model_name
    classifier_name = args.classifier_name
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

    TEST_DATA_DIR = BASE_DIR + '/classifier/'
    remove_and_create_dir(TEST_DATA_DIR)

    data_dim = data_dict['data_dim']
    n_classes = data_dict['num_clusters']
    classifier_paras = {"input_dim": data_dim, "n_classes": n_classes}
    classifier = init_classifier(classifier_name, classifier_paras)

    train_losses, classifier = classifier_train(
        classifier, data_dict, train_options)

    test_loss = classifier_test(classifier, data_dict)

    result_dict = OrderedDict()
    result_dict['classifier_name'] = classifier_name
    result_dict['classifier_paras'] = classifier_paras
    result_dict['classifier_state_dict'] = classifier.state_dict()
    result_dict['train_losses'] = train_losses
    result_dict['test_loss'] = test_loss

    write_pkl(fname = TEST_DATA_DIR + '/classifier.pkl', 
              input_dict = result_dict)
