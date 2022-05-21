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


def compute_and_save_result(epsilons, eigen_values):
        
        r = 1

        codesign_losses = []
        benchmark_losses = []
        sep_design_losses = []

        max_improv = 0

        for epsilon in epsilons:

            _, _, T = compute_optimal_scale_laplace(epsilon, r, eigen_values)
            codesign_loss = compute_optimal_loss_laplace(epsilon, r, eigen_values, T)
            codesign_losses.append(codesign_loss)

            benchmark_loss = compute_benchmark_loss_laplace(epsilon, r, eigen_values)
            benchmark_losses.append(benchmark_loss)

            sep_design_loss = compute_sep_design_loss_laplace(epsilon, r, eigen_values, sep_design_z)
            sep_design_losses.append(sep_design_loss)

            max_improv = max(max_improv, 1 - codesign_loss / benchmark_loss)
        
        print("max_improv:{}".format(max_improv))

        print("######################")
        print("co-design")
        print("######################")

        for i in range(len(epsilons)):
            print("({},{})".format(1/epsilons[i], codesign_losses[i]))


        print("######################")
        print("benchmark")
        print("######################")

        for i in range(len(epsilons)):
            print("({},{})".format(1/epsilons[i], benchmark_losses[i]))

        print("######################")
        print("separate-design")
        print("######################")

        for i in range(len(epsilons)):
            print("({},{})".format(1/epsilons[i], sep_design_losses[i]))


        # save for plotting later
        result_dict = OrderedDict()

        result_dict['epsilons'] = epsilons
        result_dict['codesign_losses'] = codesign_losses
        result_dict['benchmark_losses'] = benchmark_losses
        result_dict['sep_design_losses'] = sep_design_losses
        
        write_pkl(fname = BASE_DIR + '/results.pkl', input_dict = result_dict)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--ldp_type', type=str)
    parser.add_argument('--eigen_values', type=str)
    parser.add_argument('--sep_design_z', type=int, default=0)
    args = parser.parse_args()

    model_name = args.model_name
    LDP_type = args.ldp_type
    eigen_values = args.eigen_values.split(',')
    eigen_values = [float(eigen_value) for eigen_value in eigen_values]
    sep_design_z = args.sep_design_z

    assert(LDP_type == "Laplace")

    BASE_DIR = SCRATCH_DIR + model_name
    remove_and_create_dir(BASE_DIR)

    epsilons = [ 1 / (0.01 * i) for i in range(1, 101)]
    compute_and_save_result(epsilons, eigen_values)
                
