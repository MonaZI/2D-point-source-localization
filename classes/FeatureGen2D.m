classdef FeatureGen2D
    properties
        fcutoff;        % the cut-off frequency
        numPoint;       % the number of points
        L;              % length of the projection lines
        nodes, weights; % the sampling points and weights based on Gauss-quadrature rule
        pixelSize;      % the pixel size
        proj;           % the projection lines
        fDisc;          % discretuzed points in frequency domain
        sampleType;     % the sample type, uniform or GQ
    end
    methods
        function obj = FeatureGen2D(proj, numPoint, fcutoff, pixelSize, sampleType)
            obj.proj = proj;
            obj.numPoint = numPoint;
            obj.fcutoff = fcutoff;
            obj.L = size(proj, 1);
            % 4xL+1 non-uniform samples in the frequency domain from
            % 0-fcutoff
            [ obj.nodes, obj.weights ]=lgwt(1 * obj.L, 0, fcutoff);
            obj.pixelSize = pixelSize;
            
            if strcmp(sampleType, 'uniform')
                obj.fDisc = [0:fcutoff].';
                obj.weights = ones(size(obj.fDisc));
            elseif strcmp(sampleType, 'GQ')
                obj.fDisc = obj.nodes;
            end
            obj.sampleType = sampleType;
        end
        
        function [mu_est, dist_mu, u] = mean_sample(obj, maxLim)
            % Computes the sample estimation of the mean feature from the
            % projection data
            % param sampleType: the sampling type used in frequency domain,
            % sampleType=uniform: uniform sampling in frequency domain
            % sampleType=GQ: sampling and weights in frequency domain based
            % on Gauss-Legendre rule.
            % return mu_est: the estimated mean feature
            
            mean_est = mean(obj.proj, 2);
            if strcmp(obj.sampleType, 'uniform')
                mu_est = fft(ifftshift(mean_est, 1), [], 1);
                mu_est = real(mu_est(1:obj.fcutoff+1));
            elseif strcmp(obj.sampleType, 'GQ')
                k = (-floor(obj.L/2):floor(obj.L/2));
                tmp = obj.nodes * k;
                tmp = exp(-2i * pi * tmp / obj.L);
                mu_est = tmp * mean_est;
            end
            [dist_mu, u] = obj.compute_distribution(mu_est, maxLim);
        end
        
        function [C_est, dist1, u] = corr_sample(obj, n_var, sample_type, r_max)
            % Computes the sample estimation of the autocorrelation feature from the
            % projection data
            % param sampleType: the sampling type used in frequency domain,
            % sampleType=uniform: uniform sampling in frequency domain
            % sampleType=GQ: sampling and weights in frequency domain based
            % on Gauss-Legendre rule.
            % return mu_est: the estimated mean feature
            % return fDisc: the sampled points in frequency domain
            L = (size(obj.proj, 1)-1)/2;
            disc_projs = obj.proj;
            %zero-padding the signals, it seems to help for the reconstruction
%             disc_projs = [ zeros(L, size(obj.proj, 2)); obj.proj; zeros(L, size(obj.proj, 2))];
            mean_est = mean(disc_projs, 2);
            l = size(mean_est, 1); %length of the zero padded signals
            fft_disc_proj = fft(ifftshift(obj.proj, 1), [], 1);
            clear disc_projs
            abs_fft = mean(abs(fft_disc_proj).^2,2);
            clear fft_disc_proj
            
            switch sample_type
                case 'uniform'
                    C_est = abs_fft(1:obj.fcutoff+1);
                case 'GQ'
                    autocorr = ifft(real(abs_fft));
                    autocorr = fftshift(autocorr);
                    % Compute samples direct
                    k1=(-floor(l/2):floor(l/2));
                    tmp = obj.nodes*k1;
                    tmp = exp(-2*pi*1i*tmp/l);
                    C_est = tmp*autocorr;
            end
            
            C_est = real((C_est - obj.numPoint)/2); %only consider the dmn with m>n;
            
            % debiasing C
            C_est = C_est - (L+1)*(n_var);
            [dist1, u] = compute_distribution(obj, C_est, r_max);
        end
        
        function [mu, dist_mu] = mean_feature(obj, radialDist)
            % Analytical computation of mean feature
            % param radialDist: the radial distances of the point sources
            % param fDisc: the sampled points in frequency domain
            % return mu: the groundtruth mean feature (Eq(6) in the paper)
            mu = zeros(length(obj.fDisc),1);
            % generating mu
            for i = 1:length(radialDist)
                mu = mu + besselj(0, 2 * pi * obj.fDisc * radialDist(i) / (obj.pixelSize * obj.L));
            end
            dist_mu = obj.compute_distribution(mu, max(radialDist));
        end
        
        function [corr, dist_corr] = corr_feature(obj, pairDist, delta)
            % Analytical computation of autocorrelation feature
            % param pairDist: the pairwise distances of the point sources
            % param fDisc: the sampled points in frequency domain
            % return corr: the groundtruth autocorrelation feature (Eq(6) in the paper)            
            corr = zeros(length(obj.fDisc),1);
            for m = 1:length(pairDist)
                corr = corr + besselj(0, 2 * pi * obj.fDisc * pairDist(m) / (delta * obj.L));
            end
            dist_corr = obj.compute_distribution(corr, max(pairDist));
        end
        
        function [dist, u] = compute_distribution(obj, f, maxLim)
            % Computes the distributions from the features
            u = linspace(0, maxLim*1.1, 3000).'; %b1 is on the interval [0, R] for mean
            t = 2 * pi * obj.fDisc / (obj.pixelSize * size(obj.proj, 1));
            mu = real(Bessel_num_int(f, u, t, obj.weights));
            dist = mu.^2;
            dist = real(dist/sum(dist));
        end
        
    end
end
