import math
import numpy as np
from math import pi
from scipy.special import jv

class FeatureGen2D:
    def __init__(self, proj, numPoint, fcutoff, pixelSize):
        """
        Initializing the instance of the class
        :param proj: the projection lines
        :param numPoint: the number of points
        :param fcutoff: the cut-off frequency
        :param pixelSize: the pixel size
        :param sampleType: the sample type, uniform or GQ
        """
        self.proj = proj
        self.numPoint = numPoint
        self.fcutoff = fcutoff
        self.L = proj.shape[0]
        self.pixelSize = pixelSize
        # nodes, weights = lgwt(0, fcutoff, self.L)
        self.fDisc = np.arange(0, fcutoff)
        self.weights = np.ones(self.fDisc.shape)

    def mean_sample(self, maxLim):
        """
        Computes the sample estimation of the mean feature from the projection data
        :param maxLim: the maximum value for which the mean sample is computed
        :return: muEst, muDist, u: the estimation of the feature, the radial distance distribution and the sampling points of the features
        """
        muEst = np.mean(self.proj, axis=1)
        muEst = np.fft.fft(np.fft.ifftshift(muEst))
        muEst = np.real(muEst[0:int(self.fcutoff)])
        [muDist, u] = self.compute_distribution(muEst, maxLim)
        return muEst, muDist, u

    def corr_sample(self, nVar, maxLim):
        """
        Computes the sample estimation of the autocorrelation feature from the projection data
        :param n_var: the variance of the noise
        :param r_max: the maximum value for which the correlation sample is computed
        :return: corrEst, corrDist, u: the estimation of the feature, the pairwise distance distribution and the sampling points of the feature
        """
        L = (self.proj.shape[0]-1)//2
        fftDisProj = np.fft.fft(np.fft.ifftshift(self.proj, 0), axis=0)
        absFFT = np.mean(np.absolute(fftDisProj)**2, 1)
        corrEst = absFFT[0:int(self.fcutoff)]
        corrEst = np.real(corrEst-self.numPoint)/2
        # debiasing C for the noisy case
        corrEst = corrEst - (L + 1) * nVar
        corrDist, u = self.compute_distribution(corrEst, maxLim)
        return corrEst, corrDist, u

    def compute_distribution(self, f, maxLim):
        u = np.linspace(0, 1.1 * maxLim, 2000)
        t = 2 * math.pi * self.fDisc / (self.pixelSize * self.proj.shape[0])
        tmp = np.real(bessel_num_int(f, u, t, self.weights))
        # np.clip(tmp, 0., None, out=tmp)
        # tmp = tmp ** 2
        dist = np.real(tmp / np.sum(tmp))
        return dist, u

def bessel_num_int(f, u, t, w):
    """
    Computes the numerical integration after multiplying by Bessel function
    Please refer to Eq. (15) for better understanding the parameters
    :param f(t): the feature
    :param u: sample points, discretizing the final result
    :param t: discrete sampling points of f
    :param w: the weights used in approximating the integration
    :return: res, the final result
    """
    # import pdb; pdb.set_trace()
    jb = jv(0, np.expand_dims(u, 1)*np.expand_dims(t, 0))
    res = np.expand_dims(u, 1) * (np.dot(jb, np.expand_dims((t*w)*f, 1)))
    return res

