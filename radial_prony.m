% the prony method applied to the pairwise distances
clear all;
rng(1);

%parameters
k_vec = [5]; %number of points
L_vec = [100,500,1000,1500];
num_iter = 100;
R = 1; % maximum distance of the points from the center

snr_vec = 1;[1000,100,10,1,0.1,0.01]; %signal to noise ratio, %for clean data replace with snr = 'clean'
num_proj = 10^4; %number of samples
theta = linspace(0, 2*pi, num_proj).';

start_r = 10; % the starting index used for the prony method
error_r_rel = zeros(length(k_vec),length(L_vec),length(snr_vec),num_iter);
error_r_MSE = zeros(length(k_vec),length(L_vec),length(snr_vec),num_iter);

for k_ind = 1:length(k_vec)
    k = k_vec(k_ind);
    M_prony = k*10; %4 for k=5
    
    for snr_ind = 1:length(snr_vec)
        if snr_ind==1
%             snr = 'clean';
            snr = snr_vec(snr_ind);
        else
            snr = snr_vec(snr_ind);
        end
        
        % generating different point source models
        for iter = 1:num_iter
            
            % generate points
            xx = point_source_generator(k, 0.1).';
            x = xx*R;
            
            % distances to the center
            rk = sqrt(x(:, 1).^2 + x(:, 2).^2);
            X = x.';
            
            % pairwise distances
            D_x = bsxfun(@minus,(X(1,:))',X(1,:));
            D_y = bsxfun(@minus,(X(2,:))',X(2,:));
            D = sqrt(D_x.^2+D_y.^2);
            D = sort(unique(D(:)));
            D = D(2:end);
            
            for L_ind = 1:length(L_vec)
                L = L_vec(L_ind);
                L2 = 2*L; %sampling in the b direction
                fcutoff = round(1*L); %can modify this cutoff as long as its below 2L
                
                [ disc_projs, r_max, delta, n_var  ] = points_gen_proj1(X, theta, L, snr);
                
                % Generate the features
                [ F_mu, ~ ] = feature_from_proj_uniform(disc_projs, X, n_var);
                ind = [start_r:start_r+M_prony-1]';
                r_interval = ind;
                b_pairwise = F_mu(ind+1);
                
                %% checking the correctness of the approximations (as a sum of exponentials)
                M = 2*L+1;
                c_approx = zeros(length(r_interval),1);
                for nn = 1:length(rk)
                    c_approx = c_approx + ...
                        ((1-1j)*exp(2j*pi*r_interval*rk(nn)/(delta*M))+ ...
                        (1+1j)*exp( -2j*pi*r_interval*rk(nn)/(delta*M) ) )./(2*sqrt(pi*2*pi*rk(nn)*r_interval/(delta*M)));
                end
                
                [prony_mat, prony_vec] = gen_prony_mat(b_pairwise.*sqrt(r_interval),k*2);
                c = pinv(prony_mat)*prony_vec;
                c = [1;c];
                r_radial = roots((c));
                
                true_harmonics_radial = exp(2j*pi*(rk)/(delta*M));
                % extract the geometry information from the roots
                % choosing the K roots that are closest to the unit circle
                % choosing the points that are in the first half of the circle
                tmp = angle(r_radial);
                
                % changing all the angles to be between 0-2\pi
                tmp(find(tmp<0)) = tmp(find(tmp<0)) + 2*pi;
                
                t1 = (tmp<=pi);
                t2 = (tmp>=0);
                t = t1&t2;
                r_radial = r_radial(t);
                [~,I] = sort(abs(1-abs(r_radial)),'ascend');
                r_radial = r_radial(I(1:k));
                r_rec = (angle(r_radial)*delta*M)/(2*pi);
                
                error_r_MSE(k_ind,L_ind,snr_ind,iter) = norm(sort(r_rec)-sort(rk),'fro');
                error_r_rel(k_ind,L_ind,snr_ind,iter) = norm(sort(r_rec)-sort(rk),'fro')/norm(rk,'fro');
                fprintf('K = %d, L = %d, snr_ind = %d, iter_sig = %d, error_MSE(iter) = %f \n', ...
                    k, L_vec(L_ind),snr_ind,iter,error_r_MSE(k_ind,L_ind,snr_ind,iter));
            end
            
        end
    end
end

save('prony_radial_01.mat')

th = 0.1;
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


