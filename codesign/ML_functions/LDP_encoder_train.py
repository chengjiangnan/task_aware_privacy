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
from train_utils import *

from textfile_utils import *
from plotting_utils import *

from collections import OrderedDict

from numpy.linalg import eig

from torch.distributions.laplace import Laplace


encoder_epochs = 15

task_agnostic_loss_fn = torch.nn.MSELoss().to(device)


def encoder_train(LDP_paras, data_dict, train_options, train_type):
    data_dim = data_dict['data_dim']
    z_dim = data_dim
    if train_type == "sep_design":
        z_dim = train_options["sep_design_z"]
    elif train_type == "codesign":
        z_dim = train_options["codesign_z"]

    if train_type == "codesign":
        encoder = init_encoder(encoder_name, {"input_dim": data_dim, "z_dim": z_dim})
        encoder_optimizer = torch.optim.Adam(
            encoder.parameters(), lr=train_options["learning_rate"], amsgrad=True, 
            weight_decay=train_options["weight_decay"])
    elif train_type == "benchmark":
        pass
    elif train_type == "sep_design":
        encoder = init_encoder(encoder_name, {"input_dim": data_dim, "z_dim": z_dim})
        encoder_optimizer = torch.optim.Adam(
            encoder.parameters(), lr=train_options["learning_rate"], amsgrad=True)
    else:
        raise Exception("undefined train_type: {}".format(train_type))
    
    decoder = init_decoder(decoder_name, {"z_dim": z_dim, "output_dim": data_dim})
    decoder_optimizer = torch.optim.Adam(
        decoder.parameters(), lr=train_options["learning_rate"], amsgrad=True)

    x = torch.tensor(data_dict['train_dataset'], dtype=torch.float32).to(device)    
    num_samples = x.shape[0]

    reference_tensor = generate_reference_tensor(data_dict, is_train=True)
    data_dict = task_precompute(data_dict, x)
    
    train_losses = []
    task_agnostic_losses = []
    
    # sep design needs to train encoder without noise first
    if train_type == "sep_design":
        print("sep_design train encoder\n")
        for i in range(train_options["num_epochs"]):
            phi = encoder(x)
            x_hat = decoder(phi)
            train_loss = compute_loss(data_dict["task"], x_hat, reference=reference_tensor)

            encoder_optimizer.zero_grad()
            train_loss.backward()
            encoder_optimizer.step()

            phi = encoder(x)
            x_hat = decoder(phi)
            train_loss = compute_loss(data_dict["task"], x_hat, reference=reference_tensor)

            decoder_optimizer.zero_grad()
            train_loss.backward()
            decoder_optimizer.step()
        print("sep_design train encoder finish\n")


    if train_type == "benchmark":
        x_np = x.detach().numpy()
        var_noise = LDP_compute_var_noise(x_np, LDP_paras)
        w = pytorch_build_noise_matrix(x_np, LDP_paras, var_noise)
    else:
        w = torch.zeros((x.shape[0], z_dim))

    for i in range(train_options["num_epochs"]):

        # for fixed w, train encoder and decoder
        for j in range(encoder_epochs):

            # train encoder only when train_type is `codesign`
            if train_type == "codesign":
                phi = encoder(x)
                x_hat = decoder(phi+w)
                train_loss = compute_loss(data_dict["task"], x_hat, reference=reference_tensor)

                encoder_optimizer.zero_grad()
                train_loss.backward()
                encoder_optimizer.step()

            # train decoder
            if train_type == "codesign" or train_type == "sep_design":
                phi = encoder(x)
            elif train_type == "benchmark":
                phi = x

            x_hat = decoder(phi+w)
            train_loss = compute_loss(data_dict["task"], x_hat, reference=reference_tensor)
            task_agnostic_loss = task_agnostic_loss_fn(x, x_hat)

            decoder_optimizer.zero_grad()
            train_loss.backward()
            decoder_optimizer.step()

        train_losses.append(train_loss.item())
        task_agnostic_losses.append(task_agnostic_loss.item())

        # update w
        phi_np = phi.detach().numpy()
        var_noise = LDP_compute_var_noise(phi_np, LDP_paras)
        w = pytorch_build_noise_matrix(phi_np, LDP_paras, var_noise)

        if (i + 1) % train_options["output_freq"] == 0:
            print("Epoch: {} train_loss: {}, task_agnostic_loss: {}, var_noise: {}\n"
                  .format(i+1, train_losses[-1], task_agnostic_losses[-1], var_noise))

    if train_type == "benchmark":
        return train_losses, task_agnostic_losses, None, decoder, x_hat.detach().numpy()

    return train_losses, task_agnostic_losses, encoder, decoder, x_hat.detach().numpy()


def encoder_test(LDP_paras, encoder, decoder, data_dict, train_type):
    x = torch.tensor(data_dict['test_dataset'], dtype=torch.float32).to(device)

    num_samples = x.shape[0]

    reference_tensor = generate_reference_tensor(data_dict, is_train=False)
    data_dict = task_precompute(data_dict, x)

    if train_type == "codesign" or train_type == "sep_design":
        phi = encoder(x)
    elif train_type == "benchmark":
        phi = x

    phi_np = phi.detach().numpy()
    var_noise = LDP_compute_var_noise(phi_np, LDP_paras)
    w = pytorch_build_noise_matrix(phi_np, LDP_paras, var_noise)

    x_hat = decoder(phi+w)
    
    test_loss = compute_loss(data_dict["task"], x_hat, reference=reference_tensor)
    task_agnostic_loss = task_agnostic_loss_fn(x, x_hat)

    return test_loss.item(), task_agnostic_loss.item(), x_hat.detach().numpy()


def compute_and_save_result(epsilon, delta, test_data_dir, encoder_name, 
                            decoder_name, data_dict, train_options, train_type):

    LDP_paras = {
                    "type": LDP_type,
                    "epsilon": epsilon,
                    "delta": delta,
                }

    train_losses, train_task_agnostic_losses, \
    encoder, decoder, train_recon_dataset = encoder_train(
        LDP_paras, data_dict, train_options, train_type)

    test_loss, test_task_agnostic_loss, test_recon_dataset = encoder_test(
        LDP_paras, encoder, decoder, data_dict, train_type)

    # save for plotting later
    result_dict = OrderedDict()

    result_dict['train_losses'] = train_losses
    result_dict['train_task_agnostic_losses'] = train_task_agnostic_losses

    result_dict['train_loss'] = train_losses[-1]
    result_dict['train_task_agnostic_loss'] = train_task_agnostic_losses[-1]
    result_dict['train_recon_dataset'] = train_recon_dataset

    result_dict['test_loss'] = test_loss
    result_dict['test_task_agnostic_loss'] = test_task_agnostic_loss
    result_dict['test_recon_dataset'] = test_recon_dataset

    result_dict['encoder_state_dict'] = encoder.state_dict() if encoder is not None else None
    result_dict['decoder_state_dict'] = decoder.state_dict()
    
    write_pkl(fname = test_data_dir + '/e_{}_d_{}.pkl'.format(epsilon, delta), 
              input_dict = result_dict)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--train_type', type=str)
    parser.add_argument('--encoder_name', type=str)
    parser.add_argument('--decoder_name', type=str)
    parser.add_argument('--ldp_epsilons', type=str)
    parser.add_argument('--ldp_deltas', type=str)
    parser.add_argument('--ldp_type', type=str)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--codesign_z', type=int, default=0)
    parser.add_argument('--sep_design_z', type=int, default=0)

    args = parser.parse_args()

    model_name = args.model_name
    train_type = args.train_type
    encoder_name = args.encoder_name
    decoder_name = args.decoder_name
    LDP_epsilons = args.ldp_epsilons.split(',')
    LDP_epsilons = [float(ldp_epsilon) for ldp_epsilon in LDP_epsilons]
    LDP_deltas = args.ldp_deltas.split(',')
    LDP_deltas = [float(ldp_delta) for ldp_delta in LDP_deltas]
    LDP_type = args.ldp_type
    num_epochs = args.num_epochs
    weight_decay = args.weight_decay
    codesign_z = args.codesign_z
    sep_design_z = args.sep_design_z

    assert(len(LDP_epsilons) == len(LDP_deltas))

    BASE_DIR = SCRATCH_DIR + model_name
    DATA_DIR = BASE_DIR + '/dataset/'
    data_dict = load_pkl(DATA_DIR + '/data.pkl')

    TEST_DATA_DIR = BASE_DIR + '/LDP_{}/'.format(train_type)
    remove_and_create_dir(TEST_DATA_DIR)

    data_dict = task_init(data_dict, BASE_DIR)

    train_options = { 
                      "num_epochs": num_epochs,
                      "learning_rate": 1e-3,
                      "output_freq": 10,
                      "save_model": True,

                      "weight_decay": weight_decay,

                      "codesign_z": codesign_z,
                      "sep_design_z": sep_design_z
                    }

    for i in range(len(LDP_epsilons)):
        epsilon = LDP_epsilons[i]
        delta = LDP_deltas[i]

        print('################')
        print('[LDP parameters] epsilon:{}, delta:{}\n'.format(epsilon, delta))

        compute_and_save_result(epsilon, delta, TEST_DATA_DIR, 
                                encoder_name, decoder_name, data_dict, train_options, train_type)       
