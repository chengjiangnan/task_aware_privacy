import torch
import torch.nn as nn
import numpy as np
import scipy.stats as st

torch.manual_seed(0)

from models.encoder_decoder import *
from models.classifier import *
from models.regressor import *

from scipy.spatial.distance import cdist
from torch.distributions.laplace import Laplace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_encoder(encoder_name, paras):
    if encoder_name == "simple_encoder":
        encoder = SimpleEncoder(paras["input_dim"], paras["z_dim"]).to(device)
    elif encoder_name == "DNN_encoder":
        encoder = DNNEncoder(paras["input_dim"], paras["z_dim"]).to(device)
    else:
        raise Exception("undefined encoder: {}".format(encoder_name))

    return encoder


def init_decoder(decoder_name, paras):
    if decoder_name == "simple_decoder":
        decoder = SimpleDecoder(paras["z_dim"], paras["output_dim"]).to(device)
    elif decoder_name == "DNN_decoder":
        decoder = DNNDecoder(paras["z_dim"], paras["output_dim"]).to(device)
    else:
        raise Exception("undefined decoder: {}".format(decoder_name))

    return decoder


def init_classifier(classifier_name, paras):
    if classifier_name == "linear_classifier":
        classifier = LinearClassifier(paras["input_dim"], paras["n_classes"]).to(device)
    elif classifier_name == "DNN_classifier":
        classifier = DNNClassifier(paras["input_dim"], paras["n_classes"]).to(device)
    elif classifier_name == "CNN_classifier":
        classifier = CNNClassifier(paras["input_dim"], paras["n_classes"]).to(device)
    else:
        raise Exception("undefined classifier: {}".format(classifier_name))

    return classifier


def init_regressor(regressor_name, paras):
    if regressor_name == "linear_regressor":
        regressor = LinearRegressor(paras["input_dim"]).to(device)
    elif regressor_name == "DNN_regressor":
        regressor = DNNRegressor(paras["input_dim"]).to(device)
    else:
        raise Exception("undefined regressor: {}".format(regressor_name))

    return regressor


# compute the sensitivity of a multi-dimensional function
# based on sample outputs 
def compute_sensitivity(latent_repre, order):
    # D is (num_samples, num_samples) distance matrix
    D = cdist(latent_repre, latent_repre, 'minkowski', order)
    sen = np.amax(D)

    return sen


# compute the variance of noise variation
# latent_representation is (num_samples, num_features) matrix
def LDP_compute_var_noise(latent_repre, LDP_paras):
    if LDP_paras["type"] == "Laplace":
        sen = compute_sensitivity(latent_repre, 1)
        var_noise = (sen / LDP_paras["epsilon"]) ** 2
    else:
        raise Exception("undefined LDP type: {}".format(LDP_paras["type"]))

    return var_noise


class Dataset(torch.utils.data.Dataset):
  def __init__(self, inputs, labels):
        self.labels = labels
        self.inputs = inputs

  def __len__(self):
        return len(self.inputs)

  def __getitem__(self, index):
        x = self.inputs[index]
        y = self.labels[index]

        return x, y


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, h
