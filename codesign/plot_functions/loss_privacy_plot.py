import numpy as np
import os, sys
import pandas
import argparse

import torch

LDP_CODESIGN_DIR = os.environ['LDP_CODESIGN_DIR'] 
sys.path.append(LDP_CODESIGN_DIR)
sys.path.append(LDP_CODESIGN_DIR + '/utils/')
sys.path.append(LDP_CODESIGN_DIR + '/codesign/')

SCRATCH_DIR = LDP_CODESIGN_DIR + '/scratch/'

from textfile_utils import *
from plotting_utils import *
from collections import OrderedDict
from utils import *
from ML_functions.train_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alg_var = r"Approach"
loss_var = r"Task Loss"
privacy_var = None


def plot_loss(data_dict):
    # for loss_type in [ 'test' ]:
    for loss_type in [ 'train', 'test' ]:

        print("################")
        print(loss_type)
        print("################")

        is_train = loss_type == "train"
        # try:

        loss_results_df = pandas.DataFrame(columns = [privacy_var, loss_var, alg_var])

        # for train_type in ['benchmark', 'sep_design']:
        # for train_type in ['codesign', 'sep_design']:
        for train_type in ['codesign', 'benchmark', 'sep_design']:
            SUB_DIR = BASE_DIR + '/LDP_{}/'.format(train_type)

            print(train_type)

            for i in range(len(LDP_epsilons)):
                epsilon = LDP_epsilons[i]
                delta = LDP_deltas[i]
                label = 'e_{}_d_{}'.format(epsilon, delta)

                privacy = 1 / epsilon
            
                pkl_file = "{}.pkl".format(label)
                result_dict = load_pkl(SUB_DIR + '/' + pkl_file)

                # this computes the mean loss
                # loss = result_dict['{}_loss'.format(loss_type)]

                x = data_dict['{}_dataset'.format(loss_type)]
                x = torch.tensor(x, dtype=torch.float32).to(device)
                x_hat = result_dict['{}_recon_dataset'.format(loss_type)]
                x_hat = torch.tensor(x_hat, dtype=torch.float32).to(device)

                data_dict = task_precompute(data_dict, x)

                reference_tensor = generate_reference_tensor(data_dict, is_train=is_train)

                # this computes the mean loss
                # loss = compute_loss(data_dict['task'], x_hat, reference=reference_tensor).item()
                # loss_results_df.loc[len(loss_results_df.index)] = [privacy, loss, train_type]

                loss = compute_loss(data_dict['task'], x_hat, reference=reference_tensor)
                loss = loss.detach().numpy()

                if loss.ndim > 1:
                    num_samples, num_dimensions = loss.shape[0], loss.shape[1]
                else:
                    num_samples, num_dimensions = loss.shape[0], 1

                loss = loss.reshape(num_dimensions * num_samples)

                basic_results_df = pandas.DataFrame()
                basic_results_df[privacy_var] = [ privacy ] * (num_dimensions * num_samples)
                basic_results_df[loss_var] = loss
                basic_results_df[alg_var] = [ train_type ] * (num_dimensions * num_samples)

                loss_results_df = loss_results_df.append(basic_results_df)

                m, h = mean_confidence_interval(loss)
                print("{} {} {}".format(privacy, m, h))

                # print("({},{})".format(privacy,loss))

        # the following statistics only works for average loss

        print("################")
        print("{}, statistics".format(loss_type))
        print("################")

        max_improv_1 = 0
        epsilon_1 = 0
        max_improv_2 = 0
        epsilon_2 = 0

        for i in range(len(LDP_epsilons)):

            losses = {}
            for train_type in ['codesign', 'benchmark', 'sep_design']:
                SUB_DIR = BASE_DIR + '/LDP_{}/'.format(train_type)

            
                epsilon = LDP_epsilons[i]
                delta = LDP_deltas[i]
                label = 'e_{}_d_{}'.format(epsilon, delta)

                privacy = 1 / epsilon
            
                pkl_file = "{}.pkl".format(label)
                result_dict = load_pkl(SUB_DIR + '/' + pkl_file)

                loss = result_dict['{}_loss'.format(loss_type)]
                losses[train_type] = loss

            curr_improv_1 = 1-losses['codesign']/losses['benchmark']
            curr_improv_2 = 1-losses['codesign']/losses['sep_design']

            # print("[epsilon {}] codesign: {}, benchmark: {}, improv: {}"
            #       .format(epsilon, losses['codesign'], losses['benchmark'], curr_improv))

            if curr_improv_1 > max_improv_1:
                max_improv_1 = curr_improv_1
                epsilon_1 = LDP_epsilons[i]
            if curr_improv_2 > max_improv_2:
                max_improv_2 = curr_improv_2
                epsilon_2 = LDP_epsilons[i]
            # max_improv_1 = max(max_improv_1, curr_improv_1)

        print("max_improv_1: {}, epsilon_1: {}\n".format(max_improv_1, epsilon_1))
        print("max_improv_2: {}, epsilon_2: {}\n".format(max_improv_2, epsilon_2))

            
        # for i in range(len(LDP_epsilons)):
        #     epsilon = LDP_epsilons[i]
        #     privacy = 1 / epsilon
        #     curr_df = loss_results_df[loss_results_df[privacy_var] == privacy]
        #     curr_improv = 1-curr_df.iloc[0][loss_var]/curr_df.iloc[1][loss_var]
        #     max_improv = min(max_improv, curr_improv)
        #     print("privacy: {}, improv: {}".format(privacy, curr_improv))
        # print("max_improv: {}\n".format(max_improv))

        plot_file = PLOT_DIR + '/{}_{}_privacy.{}'.format(model_name, loss_type, fig_ext)

        sns.set(font_scale = 1.5)
        sns.lineplot(x=privacy_var, y=loss_var, data=loss_results_df, 
                     hue=alg_var, lw=3, marker='o', markersize=10)

        # plt.xscale('log')
        plt.legend()
        # plt.title(loss_type + ',' + LDP_type)
        # plt.title(LDP_type)
        plt.savefig(plot_file)
        plt.close()

        # except Exception as e:
        #     print(e)
        #     continue


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--ldp_epsilons', type=str, default="")
    parser.add_argument('--ldp_deltas', type=str, default="")
    parser.add_argument('--ldp_type', type=str)
    parser.add_argument('--fig_ext', type=str)
    args = parser.parse_args()

    model_name = args.model_name
    LDP_epsilons = args.ldp_epsilons.split(',')
    LDP_epsilons = [float(ldp_epsilon) for ldp_epsilon in LDP_epsilons]
    LDP_deltas = args.ldp_deltas.split(',')
    LDP_deltas = [float(ldp_delta) for ldp_delta in LDP_deltas]
    LDP_type = args.ldp_type
    fig_ext = args.fig_ext

    privacy_var = r"LDP Budget $1 / \epsilon$"

    BASE_DIR = SCRATCH_DIR + model_name
    PLOT_DIR = BASE_DIR + '/loss_privacy_plot/'
    remove_and_create_dir(PLOT_DIR)

    DATA_DIR = BASE_DIR + '/dataset/'
    data_dict = load_pkl(DATA_DIR + '/data.pkl')

    data_dict = task_init(data_dict, BASE_DIR, reduction='none')

    plot_loss(data_dict)
