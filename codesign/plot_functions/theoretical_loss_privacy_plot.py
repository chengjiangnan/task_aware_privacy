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


def plot_loss(result_dict):

        privacys = [1/epsilon for epsilon in result_dict['epsilons']]

        loss_results_df = pandas.DataFrame(columns = [privacy_var, loss_var, alg_var])

        for train_type in ['codesign', 'benchmark', 'sep_design']:

            basic_results_df = pandas.DataFrame()
            basic_results_df[privacy_var] = privacys
            basic_results_df[loss_var] = result_dict['{}_losses'.format(train_type)]
            basic_results_df[alg_var] = [ train_type ] * len(privacys)

            loss_results_df = loss_results_df.append(basic_results_df)

                # print("({},{})".format(privacy,loss))

        #  the following statistics only works for average loss
        # max_improv = 0
        # for i in range(len(LDP_epsilons)):
        #     epsilon = LDP_epsilons[i]
        #     privacy = 1 / epsilon
        #     curr_df = loss_results_df[loss_results_df[privacy_var] == privacy]
        #     curr_improv = 1-curr_df.iloc[0][loss_var]/curr_df.iloc[1][loss_var]
        #     max_improv = max(max_improv, curr_improv)
        #     print("privacy: {}, improv: {}".format(privacy, curr_improv))
        # print("max_improv: {}\n".format(max_improv))

        plot_file = PLOT_DIR + '/{}_loss_privacy.{}'.format(model_name, fig_ext)

        sns.set(font_scale = 1.5)
        sns.lineplot(x=privacy_var, y=loss_var, data=loss_results_df, 
                     hue=alg_var, lw=1)

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
    parser.add_argument('--fig_ext', type=str)
    args = parser.parse_args()

    model_name = args.model_name
    fig_ext = args.fig_ext

    privacy_var = r"LDP Budget $1 / \epsilon$"

    BASE_DIR = SCRATCH_DIR + model_name
    PLOT_DIR = BASE_DIR + '/loss_privacy_plot/'
    remove_and_create_dir(PLOT_DIR)

    result_dict = load_pkl(BASE_DIR + '/results.pkl')

    plot_loss(result_dict)
