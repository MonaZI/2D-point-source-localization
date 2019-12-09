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
    # plt.legend()
    plt.xlabel('u')
    plt.ylabel(feature)
    plt.savefig(feature+'.png')
