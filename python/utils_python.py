import numpy as np
from math import pi
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def plot_feature_baseline(f, r,  locs, feature='mean'):
    """
    Plots the estimated feature and compares it to the baseline

    :param f: the feature
    :param r: the radial shell
    :param locs: the baseline location of the peaks, i.e. radial or pairwise distances
    :param feature: the name of the feature
    """
    plt.figure()
    plt.plot(r, f, label='Estimated')
    plt.plot(locs, np.max(f) * np.ones(locs.shape), marker='x', linestyle='None', label='Ground truth')
    plt.legend()
    plt.xlabel('u')
    plt.ylabel(feature)
    plt.show()
    plt.savefig(feature+'.png')


def gen_prony_mat(b, K):
    """
    Generates the extended prony matrix and vector based on samples b, for K number of complex exponentials
    :param b: the samples of the sum of complex exponentials
    :param K: the number of complex exponentials
    :return: the extended prony matrix
    """

    M = len(b)

    prony_mat = np.zeros((M-K,K))
    prony_vec = np.zeros((M-K,1))
    m = np.arange(M-3, K-3, -1) #[M-2:-1:K-1]-1 = [M-3:-1:K-2];

    for k in range(len(m)):
        prony_mat[m[k]-K+2,:] = b[np.arange(m[k]+1, m[k]-K+1, -1)]#:-1:m[k]-K+2]
        prony_vec[m[k]-K+2] = -b[m[k]+2]

    # generate the extended prony matrix
    prony_mat1 = np.zeros((M-K,K))
    prony_vec1 = np.zeros((M-K,1))
    for k in range(M-K): #= 1:M-K
        prony_mat1[k,:] = np.conjugate(b[k+1:k+K+1]) #(k+1:k+K)
        prony_vec1[k] = -np.conjugate(b[k])
    prony_mat = np.concatenate((prony_mat, prony_mat1), axis=0)
    prony_vec = np.concatenate((prony_vec, prony_vec1), axis=0) #[prony_vec;prony_vec1];

    for k in range(M-K): #1:M-K
        prony_mat1[k,:] = b[np.arange(k-1+K, k-1, -1)] #[k-1+K:-1:k] #b([k-1+K:-1:k]);
        prony_vec1[k] = b[k+1]

    return prony_mat, prony_vec


def plot_roots(est_harmonics, dists, pixelSize, M, mode='radial'):
    """
    plots the harmonics corresponding to the ground truth and the estimated values for either radial or pairwise distances
    :param est_harmonics: the estimated harmonis (the output of the prony method)
    :param dists: the ground truth distances, either radial or pairwise distances
    :param pixelSize: the size of the pixel
    :param M: the dimension of the projection lines
    :param mode: either radial or pairwise
    :return:
    """
    plt.figure()
    theta1 = np.linspace(0, 2 * pi, 1000)
    plt.plot(np.cos(theta1),np.sin(theta1), label='unit circle')

    true_harmonics_radial = np.exp(2j * pi * (dists) / (pixelSize * M))
    plt.scatter(np.real(true_harmonics_radial),np.imag(true_harmonics_radial),marker='o', s=30, label='gt.')

    plt.scatter(np.real(est_harmonics / np.absolute(est_harmonics)),np.imag(est_harmonics / np.absolute(est_harmonics)),marker='x', s=60, label='est.')
    plt.legend()
    plt.title('The locations of the gt and estimated harmonics (for ' + mode +' distances) on the unit circle')
