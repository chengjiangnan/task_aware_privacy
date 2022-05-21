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


alg_var = r"Approach"
mse_var = r"Mean Squared Error"
index_var = r"Dimension $i$ in Data Sample Vector $x$"


def plot_variance():
    for dataset_type in [ 'train', 'test' ]:

        print("################")
        print(dataset_type)
        print("################")


        data_dict = load_pkl(BASE_DIR + '/dataset/data.pkl')

        original_dataset = data_dict['{}_dataset'.format(dataset_type)]

        loss_results_df = pandas.DataFrame(columns = [index_var, mse_var, alg_var])

        # try:
        for train_type in ['codesign', 'benchmark', 'sep_design']:

            print(train_type)

            SUB_DIR = BASE_DIR + '/LDP_{}/'.format(train_type)

            label = 'e_{}_d_{}'.format(epsilon, delta)
            pkl_file = "{}.pkl".format(label)
            result_dict = load_pkl(SUB_DIR + '/' + pkl_file)

            recon_dataset = result_dict['{}_recon_dataset'.format(dataset_type)]


            mse_losses = np.square(original_dataset.transpose() - recon_dataset.transpose())

            num_dimensions, num_samples = mse_losses.shape

            indices = [ int(i/num_samples)+1 for i in range(num_dimensions * num_samples)]

            basic_results_df = pandas.DataFrame()
            basic_results_df[index_var] = indices
            basic_results_df[mse_var] = mse_losses.reshape(num_dimensions * num_samples)
            basic_results_df[alg_var] = [ train_type ] * (num_dimensions * num_samples)

            loss_results_df = loss_results_df.append(basic_results_df)

            for i in range(num_dimensions):
                curr_losses = mse_losses[i, :]
                m, h = mean_confidence_interval(curr_losses)
                print("{} {} {}".format(i+1, m, h))

        # except Exception as e:
        #     print(e)
        #     continue

        # print(loss_results_df)

        plot_file = PLOT_DIR + '/{}_{}_mse.{}'.format(model_name, dataset_type, fig_ext)

        # sns.pointplot(x=index_var, y=mse_var, data=loss_results_df, hue=alg_var)
        sns.set(font_scale = 1.5)
        ax = sns.pointplot(x=index_var, y=mse_var, data=loss_results_df, hue=alg_var, 
                           err_style="bars", lw=3, marker="o")

        # plt.xscale('log')

        plt.legend()
        # plt.legend(loc="upper left")
        # plt.legend(loc="upper left")
        # plt.title(loss_type + ',' + LDP_type)
        # plt.title(LDP_type)
        # plt.xlim(0.5,num_dimensions+0.5)
        # plt.xlabel(fontsize=10)

        if num_dimensions >= 10:
            for ind, label in enumerate(ax.get_xticklabels()):
                if (ind+1) % 2 == 0:  # every 10th label is kept
                    label.set_visible(True)
                else:
                    label.set_visible(False)

        plt.savefig(plot_file)
        plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--ldp_epsilon', type=float)
    parser.add_argument('--ldp_delta', type=float)
    parser.add_argument('--fig_ext', type=str)
    args = parser.parse_args()

    model_name = args.model_name
    epsilon = args.ldp_epsilon
    delta = args.ldp_delta
    fig_ext = args.fig_ext

    BASE_DIR = SCRATCH_DIR + model_name
    PLOT_DIR = BASE_DIR + '/mse_plot/'
    remove_and_create_dir(PLOT_DIR)

    plot_variance()
