% the prony method applied to the radial distances, subsection 2.2 in the
% paper. Evaluating the method for different settings.
RunMe;
clear all; close all; clc;

% parameters of the experiment
k_vec = [5];            %number of points
L_vec = [1000, 1500];   % 2L+1: the number of discretizations of the projection lines
num_iter = 100;         % number of random realizations of point source models
numProj = 10^4;         %number of samples
R = 1;                  % the support of the signal
minDist = 0.1;          % the minimum distance
distType = 'radial';    % the distance for which we apply the constraint on the point-source model
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
                pixelSize = 3*max(points.radialDist)/(2*points.L+1);
                [disc_projs, n_var] = points.proj_1d_point(pixelSize, snr);
                
                % generate the features
                feature = FeatureGen2D(disc_projs, numPoint, fcutoff, pixelSize, sampleType);
                uMax = 1.1 * max(points.radialDist);
                [sampleMean, muSampleDist, u] = feature.mean_sample(uMax);
                
                ind = [start_r:start_r+M_prony-1]';
                r_interval = ind;
                b_pairwise = sampleMean(ind+1);
                
                M = size(disc_projs, 1);
                [prony_mat, prony_vec] = gen_prony_mat(b_pairwise .* sqrt(r_interval), numPoint * 2);
                c = pinv(prony_mat)*prony_vec;
                c = [1;c];
                r_radial = roots((c));
                
                % extract the geometry information from the roots
                tmp = angle(r_radial);
                % choosing the points that are in the first half of the circle
                ind1 = (tmp<=pi);
                ind2 = (tmp>=0);
                index = ind1 & ind2;
                
                % choosing the K roots that are closest to the unit circle
                r_radial = r_radial(index);
                [~,I] = sort(abs(1-abs(r_radial)),'ascend');
                r_radial = r_radial(I(1:numPoint));
                r_rec = (angle(r_radial) * pixelSize * M) / (2 * pi);
                
                error_r_MSE(k_ind,L_ind,snr_ind,iter) = norm(sort(r_rec)-sort(points.radialDist),'fro');
                error_r_rel(k_ind,L_ind,snr_ind,iter) = norm(sort(r_rec)-sort(points.radialDist),'fro')/norm(points.radialDist,'fro');
                fprintf('K = %d, L = %d, snr = %d, iter_sig = %d, error_MSE = %f \n', ...
                    numPoint, L_vec(L_ind), snr_vec(snr_ind), iter, error_r_MSE(k_ind,L_ind,snr_ind,iter));
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


