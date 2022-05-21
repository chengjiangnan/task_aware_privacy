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
from Linear_case_utils import *

from textfile_utils import *
from plotting_utils import *

from collections import OrderedDict

from numpy.linalg import inv, norm


def evaluate_avr_loss(
    a, projection, LDP_paras, var_noise, unit_noise_matrix, K, scaler, Omega, dataset):
    Y = K.dot(dataset)

    if projection is None:
        recon_dataset = apply_scaler(np.zeros((dataset.shape[0], dataset.shape[1])), scaler)
    else:
        scaled_projection = np.zeros(projection.shape)
        inverse_scaled_projection = np.zeros(projection.shape)
        for i in range(projection.shape[0]):
            if a[i] > 1e-6:
                scaled_projection[i, :] = a[i] * projection[i, :]
                # the following only works when projection consists of orthogonal rows
                # inverse_scaled_projection[i, :] = a[i] / (a[i]**2 + var_noise) * projection[i, :]

        # Optimal Decoder
        inverse_scaled_projection = scaled_projection.transpose().dot(scaled_projection)
        inverse_scaled_projection = inverse_scaled_projection + var_noise * np.identity(len(a))
        inverse_scaled_projection = scaled_projection.dot(inv(inverse_scaled_projection))

        scaled_dataset = apply_scaler(dataset, -scaler)
        latent_dataset = inv(Omega).dot(scaled_dataset)
        latent_repre = scaled_projection.dot(latent_dataset)

        w = unit_noise_matrix * var_noise ** 0.5

        latent_repre += w

        recon_latent_dataset = inverse_scaled_projection.transpose().dot(latent_repre)
        recon_scaled_dataset = Omega.dot(recon_latent_dataset)
        recon_dataset = apply_scaler(recon_scaled_dataset, scaler)

    Y_hat = K.dot(recon_dataset)
    return compute_mse(Y, Y_hat), Y, Y_hat, recon_dataset


def compute_var_noise(a, projection, latent_dataset, LDP_paras):
    # a is the scale vector
    # each row of the projection is a unit vector
    # dataset is the transformed dataset (not original)

    scaled_projection = np.zeros(projection.shape)
    for i in range(projection.shape[0]):
        scaled_projection[i, :] = a[i] * projection[i, :]

    latent_repre = scaled_projection.dot(latent_dataset)

    return LDP_compute_var_noise(latent_repre.transpose(), LDP_paras)


def compute_unit_projection(scaled_projection):
    a = np.zeros(scaled_projection.shape[0])
    projection = np.zeros(scaled_projection.shape)
    for i in range(scaled_projection.shape[0]):
        a[i] = norm(scaled_projection[i, :])
        projection[i, :] =  (1/a[i]) * scaled_projection[i, :]

    return a, projection


def compute_and_save_result(P, scaler, Omega, test_data_dir, data_dict, train_type):

    PCA_eigen_values, PCA_eigen_vectors = PCA(P)

    # print(PCA_eigen_values)
    # print(PCA_eigen_vectors)

    K = data_dict['task']['K']

    train_dataset = data_dict['train_dataset'].transpose()
    test_dataset = data_dict['test_dataset'].transpose()

    scaled_train_dataset = apply_scaler(train_dataset, -scaler)
    latent_train_dataset = inv(Omega).dot(scaled_train_dataset)

    scaled_test_dataset = apply_scaler(test_dataset, -scaler)
    latent_test_dataset = inv(Omega).dot(scaled_test_dataset)

    train_unit_noise = np.random.laplace(0, 1 / 2 ** 0.5, train_dataset.shape)
    test_unit_noise = np.random.laplace(0, 1 / 2 ** 0.5, test_dataset.shape)

    LDP_paras = {
                    "type": LDP_type,
                    "epsilon": None,
                    "delta": None
                }

    for i in range(len(LDP_epsilons)):
        epsilon = LDP_epsilons[i]
        delta = LDP_deltas[i]

        LDP_paras["epsilon"] = epsilon
        LDP_paras["delta"] = delta

        print('################')
        print('[LDP parameters] epsilon:{}, delta:{}\n'.format(epsilon, delta))

        # compute the scale `a` and `projection`
        if train_type == "codesign" and LDP_paras["type"] == "Laplace":
            r = compute_radius(latent_train_dataset)
            a, _, _ = compute_optimal_scale_laplace(epsilon, r, PCA_eigen_values)
            projection = PCA_eigen_vectors.transpose()

            print("a: {}\n".format(a))
        elif train_type == "benchmark":
            a, projection = compute_unit_projection(Omega)
            print("a: {}\n".format(a))
        elif train_type == "sep_design":
            a = np.zeros(len(PCA_eigen_values))
            for i in range(sep_design_z):
                a[i] = 1
            projection = PCA_eigen_vectors.transpose()

            print("a: {}\n".format(a))
        else:
            raise Exception("undefined train_type: {}".format(train_type))

        var_noise_train = compute_var_noise(a, projection, latent_train_dataset, LDP_paras)
        var_noise_test = compute_var_noise(a, projection, latent_test_dataset, LDP_paras)

        train_loss, train_Y, train_Y_hat, train_recon_dataset = evaluate_avr_loss(
            a, projection, LDP_paras, var_noise_train, train_unit_noise,
            K, scaler, Omega, train_dataset)

        test_loss, test_Y, test_Y_hat, test_recon_dataset = evaluate_avr_loss(
            a, projection, LDP_paras, var_noise_test, test_unit_noise,
            K, scaler, Omega, test_dataset)
        
        # save for plotting later
        result_dict = OrderedDict()

        result_dict['train_Y'] = train_Y
        result_dict['train_Y_hat'] = train_Y_hat
        result_dict['train_recon_dataset'] = train_recon_dataset.transpose()
        result_dict['train_loss'] = train_loss

        result_dict['test_Y'] = test_Y
        result_dict['test_Y_hat'] = test_Y_hat
        result_dict['test_recon_dataset'] = test_recon_dataset.transpose()
        result_dict['test_loss'] = test_loss

        print("train_loss: {}, test_loss: {}\n".format(train_loss, test_loss))
        
        write_pkl(fname = test_data_dir + '/e_{}_d_{}.pkl'.format(epsilon, delta), 
                  input_dict = result_dict)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--train_type', type=str)
    parser.add_argument('--ldp_epsilons', type=str)
    parser.add_argument('--ldp_deltas', type=str)
    parser.add_argument('--ldp_type', type=str)
    parser.add_argument('--sep_design_z', type=int, default=0)
    args = parser.parse_args()

    model_name = args.model_name
    train_type = args.train_type
    LDP_epsilons = args.ldp_epsilons.split(',')
    LDP_epsilons = [float(ldp_epsilon) for ldp_epsilon in LDP_epsilons]
    LDP_deltas = args.ldp_deltas.split(',')
    LDP_deltas = [float(ldp_delta) for ldp_delta in LDP_deltas]
    LDP_type = args.ldp_type
    sep_design_z = args.sep_design_z

    assert(len(LDP_epsilons) == len(LDP_deltas))

    BASE_DIR = SCRATCH_DIR + model_name
    DATA_DIR = BASE_DIR + '/dataset/'
    data_dict = load_pkl(DATA_DIR + '/data.pkl')

    TEST_DATA_DIR = BASE_DIR + '/LDP_{}/'.format(train_type)
    remove_and_create_dir(TEST_DATA_DIR)

    # find the latent variable
    train_dataset = data_dict['train_dataset'].transpose()
    scaler = scaling(train_dataset)
    Omega = latent_variable_transformation(apply_scaler(train_dataset, -scaler))

    K = data_dict['task']['K']
    P = K.dot(Omega)

    compute_and_save_result(P, scaler, Omega, TEST_DATA_DIR, data_dict, train_type)
                
