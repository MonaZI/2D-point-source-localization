% An example script for computing the rotation invariant features from the
% projection lines, Equation (15) in the paper
clear; close all; clc;

% parameters of the experiment
numPoint = 4;           % the number of points
numProj = 10e3;         % the number of projections
L = 1000;               % 2L+1 = the number of discretizations of the projection lines
fcutoff = round(1.5*L); % can modify this cutoff as long as its below 2L
snr = 'clean';          % signal to noise ratio, for clean data replace with snr = 'clean'
seed = 3;               % random seed, used for flexibility
R = 1;                  % maximum distance of the points from the center
rng(seed);

points = PointGen2D(numPoint, numProj, L, R, seed);
supp = 1.5 * max(points.pairDist);
pixelSize = supp / (2 * L + 1);
% generate the projections
sigmaG = 0.01;
[proj, n_var] = points.proj_1d_point_gauss(pixelSize, sigmaG, snr);

% generate the features
sampleType = 'uniform';
feature = FeatureGen2D(proj, numPoint, fcutoff, pixelSize, sampleType);

% mean feature
uMax = 1.1 * max(points.radialDist);
[sampleMean, muSampleDist, u] = feature.mean_sample(uMax);
plot_feature_baseline(muSampleDist, u, points.radialDist, 'Mean');

% autocorrelation feature
uMax = 1.1 * max(points.pairDist);
[sampleCorr, distCorr, u] = feature.corr_sample(n_var, sampleType, uMax);
plot_feature_baseline(distCorr, u, points.pairDist, 'Auto-correlation');


