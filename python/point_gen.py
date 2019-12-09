import math
import numpy as np
from scipy import stats
from math import sin, cos, pi, sqrt


class PointGen2D:
    def __init__(self, args):
        """
        Initializing the instance of the class
        :param numPoint: the # of point sources
        :param numProj: the # of projections
        :param minDist: the minimum distance to contrain the generation of the point-source model
        :param distType: the distance type used, [radial, pairwise]
        :param L: 2 x L + 1 is the number of discretizations of the projection lines
        :param R: the support of the 2D point-sources
        :param seed: the seed
        :param constrained: whether the points are generated based on the minimum distance constraint or not
        """
        np.random.seed(args.seed)
        self.numPoint = args.numPoint
        self.numProj = args.numProj
        self.R = args.R
        self.L = args.L
        if (not args.constrained):
            locs = self.gen_point_source()
        else:
            locs = self.gen_point_source_constrained(args.minDist, args.distType)
        self.X = locs[:, 0] - np.mean(locs[:, 0])
        self.Y = locs[:, 1] - np.mean(locs[:, 1])
        self.theta = np.linspace(0, 2 * math.pi, num=args.numProj)
        self.radialDist = self.radial_distance()
        self.pairDist = self.pair_distance()

    def gen_point_source(self):
        """
        Generates the locations of the point sources
        :return: pts_loc, coordinates of the point source locations
        """
        pts_loc = self.R * (1./np.sqrt(2)) * (2 * np.random.uniform(size=(self.numPoint, 2)) - 1)
        return pts_loc

    def gen_point_source_constrained(self, minDist, distType):
        """
        generates the point-source model with #numPoint points with such that the minimum (either radial or pairwise)
        distance between the distances is minDist.
        :param minDist: the minimum distance
        :param distType: the distance for which we have the minimum distance
        :return: coord, the coordinates of the points
        """
        coord = []
        trial = 1
        while(coord.shape[0] < self.numPoint):
            iter = 1
            coord = np.random.uniform(size=(2, 1))
            while (iter < 1000) and (len(coord) < self.numPoint):
                newPoint = np.random.uniform(size=(2, 1)) - 0.5
                if distType=='radial':
                    radialNew = np.linalg.norm(newPoint)
                    radialDist = np.sqrt(np.sum(coord**2, 1))
                    diffNewDist = np.absolute(radialDist - radialNew)
                else:
                    coordTmp = np.concatenate(coord, newPoint, axis=1)
                    Dx = coordTmp[:, 1] - np.transpose(coordTmp[:, 1])
                    Dy = coordTmp[:, 2] - np.transpose(coordTmp[:, 2])
                    D = np.sqrt(Dx**2 + Dy**2)
                    D = np.sort(np.unique(D.reshape(-1, 1)))
                    D = D[1:]
                    diffNewDist = []
                    for k1 in range(len(D)):
                        for k2 in range(len(D)):
                            diffNewDist.append(abs(D_oracle(k1) - D_oracle(k2)))
                    diffNewDist = np.array(diffNewDist)

                if (not np.where(diffNewDist < minDist)):
                    coord = np.concatenate(coord, newPoint, axis=1)

                iter += 1
            trial +=1
        return coord

    def radial_distance(self):
        """
        Computes the distance from the origin of the given point sources
        :return: r, the radial distances
        """
        r = np.sqrt(np.sum(np.expand_dims(self.X**2, 1) + np.expand_dims(self.Y**2, 1), 1))
        return r

    def pair_distance(self):
        """
        Computes the pairwise distance between the given point sources
        :return: d, the sorted pairwise distances
        """
        # import pdb; pdb.set_trace()
        xx = np.expand_dims(self.X, 1)
        yy = np.expand_dims(self.Y, 1)
        Dx = xx - np.transpose(xx)
        Dy = yy - np.transpose(yy)
        D = np.sqrt(Dx**2 + Dy**2)
        D = np.sort(np.unique(D.reshape(-1, 1)))
        D = D[1:]
        return D

    def proj_1d_point_gauss(self, pixelSize, sigma, snr):
        """
        generates random 1D projections of the point source model.
        here a Gaussian is convolved with the point source model
        :param pixelSize: the sampling step of the projection lines
        :param sigma: the std of the Gaussian signal used to generate the projection lines
        :param snr: the signal-to-noise ration of the projection lines
        :return proj, nVar: the projections and the variance of the noise
        """
        rX = np.dot(np.expand_dims(self.X, 1), np.sin(np.expand_dims(self.theta, 0))) - \
             np.dot(np.expand_dims(self.Y, 1), np.cos(np.expand_dims(self.theta, 0)))
        rX = np.round(rX / pixelSize)
        rows = rX + self.L + 1
        gaussWidth = 6 * np.floor(sigma / pixelSize) + 1
        halfWidth = int(3 * np.floor(sigma / pixelSize))
        gridGauss = np.arange(-3 * np.floor(sigma/pixelSize), 3 * np.floor(sigma/pixelSize) + 1) * pixelSize
        gaussSignal = 1/(np.sqrt(2 * pi * sigma**2)) * np.exp(-gridGauss**2 / (2 * sigma**2))
        M = 2 * self.L # length of the 1D projection line
        proj = np.zeros((M+1, self.numProj))
        for k in range(self.numProj):
            print(k)
            for n in range(self.numPoint):
                if rows[n, k] < halfWidth:
                    proj[0:int(rows[n, k]+halfWidth), k] += gaussSignal[int(halfWidth-rows[n, k] + 1):]
                elif (rows[n, k]==halfWidth):
                    # import pdb; pdb.set_trace()
                    proj[0:int(rows[n, k] + halfWidth)+1, k] += gaussSignal
                elif ((rows[n, k] + halfWidth) > M):
                    proj[int(rows[n, k] - halfWidth):, k] += gaussSignal[0:int(gaussWidth-(rows[n, k]+halfWidth-M))]
                elif ((rows[n, k] + halfWidth) == M):
                    proj[int(rows[n, k]-halfWidth):, k] += gaussSignal
                elif (rows[n, k]>halfWidth) and (rows[n, k]+halfWidth<M):
                    proj[int(rows[n, k] - halfWidth):int(rows[n, k] + halfWidth +1), k] += gaussSignal
        sig_var = np.var(proj.reshape(-1, 1))
        nVar = 0
        if snr!='clean':
            nVar = (1/snr) * sig_var  # the variance of the noise
            proj = proj + np.sqrt(nVar) * np.random.normal(size=proj.shape)

        #TODO: you should perform zero-padding here if u want
        proj = np.concatenate((np.zeros((self.L, self.numProj)), proj, np.zeros((self.L, self.numProj))), axis = 0)
        # disc_projs = [ zeros(obj.L, size(disc_projs, 2)); disc_projs; zeros(obj.L, size(disc_projs, 2))];

        return proj, nVar

    def proj_1d_point(self, pixelSize, snr):
        """
        Generates the projections form the point source model (with no Gaussian kernel)
        :param pixelSize: the sampling step of the projection lines
        :param snr: the signal-to-noise ration of the projection lines
        :return proj, nVar: the projections and the variance of the noise
        """
        rX = np.dot(np.expand_dims(self.X, 1), np.sin(np.expand_dims(self.theta, 0))) - \
             np.dot(np.expand_dims(self.Y, 1), np.cos(np.expand_dims(self.theta, 0)))
        rX = np.round(rX / pixelSize)
        locs = (rX + self.L ).astype(int)
        cols = np.tile(np.arange(0, self.numProj, dtype=int), (self.numPoint,1))
        import pdb; pdb.set_trace()
        proj = np.zeros((2*self.L+1, self.numProj))
        proj[locs.reshape(-1, 1), cols.reshape(-1,1)] = 1.
        sig_var = np.var(proj.reshape(-1, 1))
        nVar = 0
        if snr!='clean':
            nVar = (1/snr) * sig_var  # the variance of the noise
            proj = proj + np.sqrt(nVar) * np.random.normal(size=proj.shape)
        return proj, nVar
