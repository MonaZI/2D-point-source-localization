% An example script for computing the rotation invariant features from the
% projection lines, Equation (15) in the paper
RunMe;
clear; close all; clc;

% parameters of the experiment
numPoint = 5;           % the number of points
numProj = 10e3;         % the number of projections
L = 1500;               % 2L+1 = the number of discretizations of the projection lines
fcutoff = round(1.5*L); % can modify this cutoff as long as its below 2L
snr = 'clean';          % signal to noise ratio, for clean data replace with snr = 'clean'
seed = 3;               % random seed, used for flexibility
R = 1;                  % maximum distance of the points from the center
rng(seed);
constrained = false;    % generate the point sources based on minimum distance constraint
minDist = 0.1;          % the minumum distance
sigmaG = 0.005;         % std of the Gaussian used in generating the Gaussian source model
distType = 'radial';
sampleType = 'GQ'; % how to sample the frequency domain to compute the features
                        % options:unifom/GQ, GQ corresponds to the sampling
                        % and integration based on Gauss-Legendre rule. To
                        % use this sample type, install ASPIRE package in
                        % http://spr.math.princeton.edu/.

points = PointGen2D(numPoint, numProj, minDist, distType, L, R, seed, constrained);
supp = 3 * max(points.pairDist);
pixelSize = supp / (2 * L + 1);
% generate the projections
[proj, n_var] = points.proj_1d_point_gauss(pixelSize, sigmaG, snr);

% generate the features
feature = FeatureGen2D(proj, numPoint, fcutoff, pixelSize, sampleType);

% mean feature
uMax = 1.1 * max(points.radialDist);
[sampleMean, muSampleDist, u] = feature.mean_sample(uMax);
plot_feature_baseline(muSampleDist, u, points.radialDist, 'Mean');

% autocorrelation feature
uMax = 1.1 * max(points.pairDist);
[sampleCorr, distCorr, u] = feature.corr_sample(n_var, sampleType, uMax);
plot_feature_baseline(distCorr, u, points.pairDist, 'Auto-correlation');


