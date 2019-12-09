import argparse
import numpy as np

from point_gen import PointGen2D
from feature_gen import FeatureGen2D
from utils import plot_feature_baseline


def main(args):
    np.random.seed(args.seed)
    fcutoff = np.round(2 * args.L)
    points = PointGen2D(args)
    supp = 2 * np.max(points.radialDist)
    pixelSize = supp / (2 * args.L + 1)
    proj, nVar = points.proj_1d_point_gauss(pixelSize, args.sigmaG, args.snr)

    # generate the features
    feature = FeatureGen2D(proj, args.numPoint, fcutoff, pixelSize)
    # mean feature
    uMax = 1.1 * np.max(points.radialDist)
    _, muSampleDist, u = feature.mean_sample(uMax)
    plot_feature_baseline(muSampleDist, u, points.radialDist, 'Mean')
    # auto-correlation feature
    uMax = 1.1 * np.max(points.pairDist)
    _, distCorr, u = feature.corr_sample(nVar, uMax)
    plot_feature_baseline(distCorr, u, points.pairDist, 'Auto-correlation')


def arg_parse():
    """
    Parses the passed arguments

    :return: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-numPoint', type=int, default=3, help='the number of points')
    parser.add_argument('-numProj', type=int, default=5000, help='the number of projection lines')
    parser.add_argument('-distType', type=str, default='radial', help='the distance use to constrain the generation of the point source model')
    parser.add_argument('-constrained', action='store_true', default=False, help='to apply the minimum separation constraint to generation of the point sources or not')
    parser.add_argument('-L', type=int, default=1000, help='the number of discretizations of the projection lines')
    parser.add_argument('-R', type=float, default=1., help='the maximum distance of the points from the center')
    parser.add_argument('-sigmaG', type=float, default=0.01, help='the std of the Gaussians used to generate the point sources')
    parser.add_argument('-snr', type=float, default=1000, help='the signal to noise ratio')
    parser.add_argument('-seed', type=int, default=1, help='the random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    main(args)
