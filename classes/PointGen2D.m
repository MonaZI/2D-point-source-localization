classdef PointGen2D
    % This class defines all the attributes and methods used for 2D point
    % sources.
    properties (Access = public)
        numPoint;
        X, Y;       % X and Y coordinates of the point-sources
        theta;      % the projection angles
        numProj;    % the number of projection images
        R;          % maximum distance of the points from the origin
        L;          % 2L+1 = the number of discretizations of the projection lines
        radialDist; % the radial distances
        pairDist;   % the pairwise distances
    end
    
    methods
        
        function obj = PointGen2D(numPoint, numProj, L, R, seed)
            rng(seed);
            obj.numPoint = numPoint;
            obj.numProj = numProj;
            obj.R = R;
            obj.L = L;
            
            % when the class is instantiated, the point sources are
            % generated.
            locs = obj.gen_point_source();
            obj.X = locs(:, 1) - mean(locs(:, 1));
            obj.Y = locs(:, 2) - mean(locs(:, 2));
            % uniformly distributed angles between [0, 2 pi]
%             obj.theta = 2 * pi * rand(numProj, 1);
            obj.theta = linspace(0, 2*pi, numProj).'; 

            
            obj.radialDist = obj.radial_distance();
            obj.pairDist = obj.pair_distance();
        end
        
        function pts_loc = gen_point_source(obj)
            % Generates the locations of 2D point sources
            % return pts_loc: coordinates of the point source locations
            
            pts_loc = obj.R*1/sqrt(2)*(2*rand(obj.numPoint, 2) - 1);
        end
        
        function r = radial_distance(obj)
            % Computes the distance from the origin of the given point sources
            % return r: the radial distances
            
            r = sqrt(obj.X.^2 + obj.Y.^2);
        end
        
        function d = pair_distance(obj)
            % Computes the pairwise distance between the point sources
            % return d: the sorted pairwise distances
            
            d = sqrt(bsxfun(@minus, obj.X, obj.X.').^2+bsxfun(@minus, obj.Y, obj.Y.').^2);
            d = sort((d(:)), 'ascend');
%             d = d(2:end);
        end
        
        function [disc_projs, n_var] = proj_1d_point_gauss(obj, pixelSize, sigma, snr)
            % generates random 1D projections of the point source model
            % here a Gaussian is convolved with the point source model
            % param pixelSize: the sampling step of the projection lines
            % param snr: the signal to noise ratio of the projection lines
            % return disc_projs: the discretized projection lines
            % return n_var: the standard deviation of noise
            
            rX = obj.X * sin(obj.theta.') - obj.Y * cos(obj.theta.');
            rX = round(rX / pixelSize).';
            % the projection lines are sparse, thus they are saved as a
            % sparse matrix to save space.
            % the locations at which the projection lines are not zero
            rows = rX + obj.L + 1;
            gaussWidth = 6 * floor(sigma/pixelSize) + 1;
            gridGauss = [-3 * floor(sigma/pixelSize):3 * floor(sigma/pixelSize)].' * pixelSize;
            gaussSignal = 1/(sqrt(2*pi*sigma^2)) * exp(-gridGauss.^2/(2*sigma^2));
            M = 2 * obj.L + 1; % length of the 1D projection line
            disc_projs = zeros(M, obj.numProj);

            for k = 1:obj.numProj
                for n = 1:obj.numPoint
                    if rows(k, n)<floor(gaussWidth/2)
                        disc_projs(1:rows(k, n)+floor(gaussWidth/2), k) = ...
                            + disc_projs(1:rows(k, n)+floor(gaussWidth/2), k) + ...
                            gaussSignal(2+floor(gaussWidth/2)-rows(k, n):end);
                    elseif (rows(k, n)==floor(gaussWidth/2))
                        disc_projs(1:rows(k, n)+floor(gaussWidth/2), k) = ...
                            + disc_projs(1:rows(k, n)+floor(gaussWidth/2), k) + ...
                            gaussSignal(2:end);
                    elseif ((rows(k, n) + floor(gaussWidth/2)) > M)
                        disc_projs(rows(k, n)-floor(gaussWidth/2):end, k) = ...
                            disc_projs(rows(k, n)-floor(gaussWidth/2):end, k) + ...
                            gaussSignal(1:gaussWidth-(rows(k, n)+floor(gaussWidth/2)-M));
                    elseif ((rows(k, n) + floor(gaussWidth/2)) == M)
                        disc_projs(rows(k, n)-floor(gaussWidth/2):end, k) = ...
                            disc_projs(rows(k, n)-floor(gaussWidth/2):end, k) + ...
                            gaussSignal(1:gaussWidth-(rows(k, n)+floor(gaussWidth/2)-M));
                    elseif (rows(k, n)>floor(gaussWidth/2)) && (rows(k, n)+floor(gaussWidth/2)<M)
                        disc_projs(rows(k, n)-floor(gaussWidth/2):rows(k, n)+floor(gaussWidth/2), k) = ...
                            disc_projs(rows(k, n)-floor(gaussWidth/2):rows(k, n)+floor(gaussWidth/2), k) + gaussSignal;
                    end
                end
            end
            sig_var = var(disc_projs(:));
            n_var = 0;
            if ~(strcmp(snr, 'clean'))
                n_var = (1/snr) * sig_var;  % the variance of the noise
                disc_projs = disc_projs + sqrt(n_var)*randn(size(disc_projs));
            end
            % zero padding the signal, to increase the resolution in
            % frequency domain
            disc_projs = [ zeros(obj.L, size(disc_projs, 2)); disc_projs; zeros(obj.L, size(disc_projs, 2))];
        end
        
        
    end
end
