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


data_dim_var = r"Data Dimension"
value_var = r"Value"
tag_var = r"Tag"

def plot_data():
    for data_type in [ 'train', 'test' ]:

        data_df = pandas.DataFrame(columns = [data_dim_var, value_var, tag_var])

        dataset = data_dict[data_type + '_dataset']
        tags = data_dict[data_type + '_tags']
        num_samples = dataset.shape[0]
        data_dim = data_dict['data_dim']

        data_df[data_dim_var] = list(range(data_dim)) * num_samples
        data_df[value_var] = dataset.reshape(num_samples * data_dim)
        data_df[tag_var] = [tag for tag in tags for i in range(data_dim)]

        # print(loss_results_df)
        plot_file = PLOT_DIR + '/{}_{}_data.{}'.format(model_name, data_type, fig_ext)

        # sns.lineplot(x=data_dim_var, y=value_var, data=data_df, hue=tag_var)
        
        sns.lineplot(x=data_dim_var, y=value_var, data=data_df, hue=tag_var, 
                     palette=sns.color_palette(n_colors=data_dict['num_clusters']))
        
        plt.legend()
        plt.title(data_type)
        plt.savefig(plot_file)
        plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--fig_ext', type=str)
    args = parser.parse_args()

    model_name = args.model_name
    fig_ext = args.fig_ext

    BASE_DIR = SCRATCH_DIR + model_name
    SUB_DIR = BASE_DIR + '/dataset/'
    PLOT_DIR = SUB_DIR + 'data_plot'
    remove_and_create_dir(PLOT_DIR)

    data_dict = load_pkl(SUB_DIR + 'data.pkl')

    plot_data()

    
