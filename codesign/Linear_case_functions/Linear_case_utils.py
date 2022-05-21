import torch
import torch.nn as nn
import numpy as np

from numpy.linalg import eigh, norm

from sklearn.metrics import mean_squared_error


def compute_mse(X, X_hat):
    # multiply by the vector length
    # return mean_squared_error(X, X_hat) * X.shape[0]
    return mean_squared_error(X, X_hat)


def scaling(dataset):
    scaler = np.mean(dataset, axis=1)
    scaler = scaler.reshape((scaler.shape[0], 1))
    return scaler


def apply_scaler(dataset, scaler):
    return dataset + scaler


def latent_variable_transformation(X):
    # transform vector x to a latent variable h
    # x = Omega * h

    data_dim = X.shape[0]
    num_samples = X.shape[1]

    eigen_values, eigen_vectors = eigh(X.dot(X.transpose())/num_samples)
    
    print(eigen_values)

    eigen_values_root = np.zeros(data_dim)
    for i in range(data_dim):
        if eigen_values[i] > 0:
            eigen_values_root[i] = eigen_values[i] ** 0.5

    Omega = eigen_vectors.dot(np.diag(eigen_values_root))

    return Omega


def PCA(P1):
    Psi = P1.transpose().dot(P1)

    # eigen decomposition
    eigen_values, eigen_vectors = eigh(Psi)
    idx = eigen_values.argsort()[::-1]

    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    return eigen_values, eigen_vectors


def contrastive_PCA(P1, P2, weight=0):
    Psi = P1.transpose().dot(P1) - weight * P2.transpose().dot(P2)

    # eigen decomposition
    eigen_values, eigen_vectors = eigh(Psi)
    idx = eigen_values.argsort()[::-1]

    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    return eigen_values, eigen_vectors


def compute_radius(latent_repre):
    # compute the radius of the smallest hypersphere
    # which is the maximum l2-norm of any the data points
    l2_norms = [norm(latent_repre[:, i]) for i in range(latent_repre.shape[1])]
    r = max(l2_norms)
    return r


def compute_optimal_scale_laplace(epsilon, r, eigen_values):
    # assume the eigen-values are in descending order
    # assume the l2-norm of the scales is M^0.5
    n = len(eigen_values)

    T = 0
    past_root_sum = 0
    M = 1
    var_noise = 8 * r ** 2 / epsilon **2 * M

    for i in range(n):
        eigen_value = eigen_values[i]
        curr_root = eigen_value ** 0.5
        if curr_root / (past_root_sum + curr_root) * (M + (T+1) * var_noise) < var_noise:
            break

        T += 1
        past_root_sum += curr_root

    a = np.zeros(n)
    for i in range(T):
        a[i] = eigen_values[i] ** 0.5 / past_root_sum * (M + T * var_noise) - var_noise
        a[i] = a[i] ** 0.5

    return a, var_noise, T


def compute_optimal_loss_laplace(epsilon, r, eigen_values, T):
    n = len(eigen_values)

    loss = 0
    for i in range(T):
        loss += eigen_values[i] ** 0.5

    loss = loss ** 2 * (8 * r ** 2 / epsilon ** 2) / (1 + T * 8 * r ** 2 / epsilon ** 2)

    for i in range(T, n):
        loss += eigen_values[i]

    return loss


def compute_benchmark_loss_laplace(epsilon, r, eigen_values):
    n = len(eigen_values)

    loss = 0
    for i in range(n):
        loss += eigen_values[i]

    loss = loss * (n * 8 * r ** 2 / epsilon ** 2) / (1 + n * 8 * r ** 2 / epsilon ** 2)

    return loss


def compute_sep_design_loss_laplace(epsilon, r, eigen_values, Z):
    n = len(eigen_values)

    loss = 0
    for i in range(Z):
        loss += eigen_values[i]

    loss = loss * (Z * 8 * r ** 2 / epsilon ** 2) / (1 + Z * 8 * r ** 2 / epsilon ** 2)

    for i in range(Z, n):
        loss += eigen_values[i]

    return loss
