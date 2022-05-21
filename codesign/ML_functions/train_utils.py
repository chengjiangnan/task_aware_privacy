import torch
import torch.nn as nn

from utils import *

from textfile_utils import *
from plotting_utils import *

def compute_loss(task, x_hat, reference=None):
    if task["task_name"] == "classification":
        tags = reference
        tags_hat = task["classifier"](x_hat)
        loss = task["loss_fn"](tags_hat, tags)
    elif task["task_name"] == "regression":
        y = reference
        y_hat = task["regressor"](x_hat)
        loss = task["loss_fn"](y_hat, y)

    elif task["task_name"] == "matrix_multiplication":
        x_hat_3d = x_hat.reshape(x_hat.shape[0], x_hat.shape[1], 1)
        K_x_hat = torch.matmul(task["K"], x_hat_3d)

        loss = task["loss_fn"](K_x_hat, task["K_x"])
    else:
        raise Exception("undefined task: {}".format(task["task_name"]))

    return loss


def generate_reference_tensor(data_dict, is_train):
    if is_train:
        token = "train"
    else:
        token = "test"

    reference_tensor = None

    if data_dict["task"]["task_name"] == "classification":
        reference_tensor = torch.tensor(data_dict[token +'_tags'], dtype=torch.long).to(device)
    elif data_dict["task"]["task_name"] == "regression":
        reference_tensor = torch.tensor(data_dict[token +'_outputs'], dtype=torch.float32).to(device)

    return reference_tensor


def task_init(data_dict, BASE_DIR, reduction='mean'):
    if data_dict["task"]["task_name"] == "classification":
        data_dict["task"]["loss_fn"] = torch.nn.CrossEntropyLoss(reduction=reduction).to(device)

        classifier_dict = load_pkl(BASE_DIR + '/classifier/classifier.pkl')
        classifier = init_classifier(
            classifier_dict['classifier_name'], classifier_dict['classifier_paras'])
        classifier.load_state_dict(classifier_dict["classifier_state_dict"])

        data_dict["task"]["classifier"] = classifier

    elif data_dict["task"]["task_name"] == "regression":
        data_dict["task"]["loss_fn"] = torch.nn.MSELoss(reduction=reduction).to(device)

        regressor_dict = load_pkl(BASE_DIR + '/regressor/regressor.pkl')
        regressor = init_regressor(
            regressor_dict['regressor_name'], regressor_dict['regressor_paras'])
        regressor.load_state_dict(regressor_dict["regressor_state_dict"])

        data_dict["task"]["regressor"] = regressor


    elif data_dict["task"]["task_name"] == "matrix_multiplication":
        data_dict["task"]["loss_fn"] = torch.nn.MSELoss(reduction=reduction).to(device)
        data_dict["task"]["K"] = torch.tensor(
            data_dict["task"]["K"], dtype=torch.float32).to(device)

    else:
        raise Exception("undefined task: {}".format(data_dict["task"]["task_name"]))

    return data_dict


def task_precompute(data_dict, x):
    if data_dict["task"]["task_name"] in [ "matrix_multiplication" ]:
        x_3d = x.reshape(x.shape[0], x.shape[1], 1)
        data_dict["task"]["K_x"] = torch.matmul(data_dict["task"]["K"], x_3d)

    return data_dict


def pytorch_build_noise_matrix(phi_np, LDP_paras, var_noise):
    if LDP_paras["type"] == "Laplace":
        w = Laplace(torch.zeros(phi_np.shape), torch.ones(phi_np.shape)*(var_noise/2)**0.5).sample()
    else:
        raise Exception("undefined LDP type: {}".format(LDP_paras["type"]))

    return w