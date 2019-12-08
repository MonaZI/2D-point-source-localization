% the prony method applied to the radial distances, subsection 2.2 in the
% paper
clear all; close all; clc;

% parameters of the experiment
k_vec = [5];            %number of points
L_vec = [100, 500, 1000, 1500];
num_iter = 100;         % number of random realizations of point source models
numProj = 10^4;         %number of samples
R = 1;                  % the support of the signal
minDist = 0.1;          % the minimum distance
distType = 'pair';      % the distance for which we apply the constraint on the point-source model
constrained = true;     % whether the points are generated with the minimum distance constraint or not
th = 0.1;               % the threshold used for computing success rate based on MSE
sampleType = 'uniform';


snr_vec = [sort(10.^[-2:3], 'descend')]; %signal to noise ratio, %for clean data replace with snr = 'clean'
theta = linspace(0, 2*pi, numProj).';

start_r = 10; % the starting index used for the prony method
error_r_rel = zeros(length(k_vec),length(L_vec),length(snr_vec),num_iter);
error_r_MSE = zeros(length(k_vec),length(L_vec),length(snr_vec),num_iter);

for k_ind = 1:length(k_vec)
    numPoint = k_vec(k_ind);
    M_prony = numPoint * 10; %4 for k=5
    
    for snr_ind = 1:length(snr_vec)
        snr = snr_vec(snr_ind);
        % generating different point source models
        for iter = 1:num_iter
            seed = iter;
            % generate points
            points = PointGen2D(numPoint, numProj, minDist, distType, 1, R, seed, constrained);
            coord = [points.X, points.Y].';            
            for L_ind = 1:length(L_vec)
                L = L_vec(L_ind);
                fcutoff = round(2*L); %can modify this cutoff as long as its below 2L

                points.L = L;
                pixelSize = 2.2*max(points.pairDist)/(2*points.L+1);
                [disc_projs, n_var] = points.proj_1d_point(pixelSize, snr);
                
                % generate the features
                feature = FeatureGen2D(disc_projs, numPoint, fcutoff, pixelSize, sampleType);
                uMax = 1.1 * max(points.pairDist);
                [sampleCorr, corrSampleDist, u] = feature.corr_sample(n_var, sampleType, uMax);
                
                ind = [start_r:start_r+M_prony-1]';
                r_interval = ind;
                b_pairwise = sampleCorr(ind+1);
                
                M = 2 * L + 1;
                [prony_mat, prony_vec] = gen_prony_mat(((b_pairwise)/2).*sqrt(r_interval),numPoint*(numPoint-1));
                c = pinv(prony_mat)*prony_vec;
                c = [1;c];
                r_pair = roots((c));
                
                % extract the geometry information from the roots
                tmp = angle(r_pair);
                % changing all the angles to be between 0-2\pi
                % choosing the points that are in the first half of the circle
                tmp(find(tmp<0)) = tmp(find(tmp<0)) + 2 * pi;
                ind1 = (tmp<=pi);
                ind2 = (tmp>=0);
                index = ind1 & ind2;
                
                % choosing the K roots that are closest to the unit circle
                r_pair = r_pair(index);
                [~,I] = sort(abs(1-abs(r_pair)),'ascend');
                r_pair = r_pair(I(1:numPoint * (numPoint - 1) / 2));
                r_rec = (angle(r_pair) * pixelSize * M) / (2 * pi);
                
                D = unique(points.pairDist); D = D(2:end);     
                
                error_r_MSE(k_ind,L_ind,snr_ind,iter) = norm(sort(r_rec)-sort(D),'fro');
                error_r_rel(k_ind,L_ind,snr_ind,iter) = norm(sort(r_rec)-sort(D),'fro')/norm(points.radialDist,'fro');
                fprintf('K = %d, L = %d, snr_ind = %d, iter_sig = %d, error_MSE(iter) = %f \n', ...
                    numPoint, L_vec(L_ind),snr_ind,iter,error_r_MSE(k_ind,L_ind,snr_ind,iter));
            end
        end
    end
end

% find the success rate for the clean case
success_rate = zeros(length(L_vec),length(snr_vec));
for k_ind = 1:length(k_vec)
    for L_ind = 1:length(L_vec)
        for snr_ind = 1:length(snr_vec)
            tmp = error_r_MSE(k_ind,L_ind,snr_ind,:);
            success_rate(L_ind, snr_ind) = length(find(tmp<=th))/num_iter;
        end
    end
end


