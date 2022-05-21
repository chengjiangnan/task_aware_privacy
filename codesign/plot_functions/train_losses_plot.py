import numpy as np
import os, sys
import pandas
import argparse

LDP_CODESIGN_DIR = os.environ['LDP_CODESIGN_DIR'] 
sys.path.append(LDP_CODESIGN_DIR)
sys.path.append(LDP_CODESIGN_DIR + '/utils/')
sys.path.append(LDP_CODESIGN_DIR + '/codesign/')

SCRATCH_DIR = LDP_CODESIGN_DIR + '/scratch/'

from textfile_utils import *
from plotting_utils import *
from collections import OrderedDict
from utils import *


epoch_var = r"Epoch"
loss_var = r"Loss (MSE)"
privacy_var = r"Privacy"

def plot_loss_bottleneck():
    for loss_type in [ 'train_losses', 'train_task_agnostic_losses' ]:
        try:
            loss_results_df = pandas.DataFrame(columns = [epoch_var, loss_var, privacy_var])

            for i in range(len(LDP_epsilons)):    
                epsilon = LDP_epsilons[i]
                delta = LDP_deltas[i]
                label = 'e_{}_d_{}'.format(epsilon, delta)
                label_legend = r'$\epsilon$=' + str(epsilon) + r',$\delta$=' + str(delta)

                pkl_file = "{}.pkl".format(label)
                result_dict = load_pkl(SUB_DIR + '/' + pkl_file)

                basic_results_df = pandas.DataFrame()
                basic_results_df[epoch_var] = list(range(1, num_epochs+1))
                basic_results_df[loss_var] = result_dict[loss_type]
                basic_results_df[privacy_var] = [ label_legend ] * num_epochs 

                loss_results_df = loss_results_df.append(basic_results_df)
            
            # print(loss_results_df)
            plot_file = PLOT_DIR + '/{}_{}.{}'.format(model_name, loss_type, fig_ext)

            sns.lineplot(x=epoch_var, y=loss_var, data=loss_results_df, hue=privacy_var)
            
            # plt.ylim((-1, 5))

            plt.legend()
            plt.title(loss_type)
            plt.savefig(plot_file)
            plt.close()

        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--train_type', type=str)
    parser.add_argument('--ldp_epsilons', type=str, default="")
    parser.add_argument('--ldp_deltas', type=str, default="")
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--fig_ext', type=str)
    args = parser.parse_args()

    model_name = args.model_name
    train_type = args.train_type
    num_epochs = args.num_epochs
    fig_ext = args.fig_ext
    LDP_epsilons = args.ldp_epsilons.split(',')
    LDP_epsilons = [float(ldp_epsilon) for ldp_epsilon in LDP_epsilons]
    LDP_deltas = args.ldp_deltas.split(',')
    LDP_deltas = [float(ldp_delta) for ldp_delta in LDP_deltas]

    label_length = len(LDP_epsilons)

    BASE_DIR = SCRATCH_DIR + model_name
    SUB_DIR = BASE_DIR + '/LDP_{}/'.format(train_type)
    PLOT_DIR = SUB_DIR + 'train_losses_plot'
    remove_and_create_dir(PLOT_DIR)

    plot_loss_bottleneck()
