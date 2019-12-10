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
        
        function obj = PointGen2D(numPoint, numProj, minDist, distType, L, R, seed, constrained)
            rng(seed);
            obj.numPoint = numPoint;
            obj.numProj = numProj;
            obj.R = R;
            obj.L = L;
            
            % when the class is instantiated, the point sources are
            % generated.
            if ~constrained
                locs = obj.gen_point_source();
            else
                locs = obj.gen_point_source_constrained(minDist, distType);
            end
            obj.X = locs(:, 1); % - mean(locs(:, 1));
            obj.Y = locs(:, 2); % - mean(locs(:, 2));
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
        
        function coord = gen_point_source_constrained(obj, minDist, distType)
            % generates the point-source model with #numPoint points with such that the
            % minimum (either radial or pairwise) distance between the distances is minDist.
            % param numPoint: the number of points
            % param minDist: the minimum distance
            % param distType: the distance for which we have the minimum distance
            % criterion, choices are radial and pairwise
            % return coord: the coordinates of the points
            
            coord = [];
            trial = 1;
            
            while (length(coord) < obj.numPoint)
                iter = 1;
                % the first point at the start of the trial
                coord = rand(2,1);
                o_dist = [norm(coord)];
                while (iter < 1000) && (length(coord) < obj.numPoint)
                    new_point = rand(2,1) - 0.5;
                    if strcmp(distType, 'radial')
                        o_dist = sqrt(sum(coord.^2,1));
                        new_dist = abs(o_dist-norm(new_point));
                    else
                        X_tmp = [coord, new_point];
                        D_x = bsxfun(@minus, (X_tmp(1,:))', X_tmp(1,:));
                        D_y = bsxfun(@minus, (X_tmp(2,:))', X_tmp(2,:));
                        D = sqrt(D_x.^2 + D_y.^2);
                        D = sort(unique(D(:)));
                        D_oracle = D(2:end);
                        
                        new_dist = [];
                        for k1 = 1:length(D_oracle)
                            for k2 = k1+1:length(D_oracle)
                                new_dist = [new_dist; abs(D_oracle(k1) - D_oracle(k2))];
                            end
                        end
                    end
                    
                    if length(find(new_dist < minDist))==0
                        coord = [coord, new_point];
                    end
                    iter = iter + 1;
                end
                trial = trial + 1;
            end
            coord = coord.';
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
        
        function [disc_projs, n_var] = proj_1d_point(obj, pixelSize, snr)
            rX = obj.X * sin(obj.theta.') - obj.Y * cos(obj.theta.');
            rX2 = round(rX / pixelSize).';
            % rX2 = rX/max(delta,delta1);
            rX2 = round(rX2);
            cols = repmat([ 1:obj.numProj ]', obj.numPoint, 1);
            rows = rX2(:) + obj.L + 1;
            disc_projs = sparse(rows, cols, ones(length(rows), 1), 2*obj.L + 1, obj.numProj);
            disc_projs = full(disc_projs);
            sig_var = var(disc_projs(:));
            n_var = 0;
            if ~(strcmp(snr, 'clean'))
                n_var = (1/snr) * sig_var;  % the variance of the noise
                disc_projs = disc_projs + sqrt(n_var)*randn(size(disc_projs));
            end
            
        end
        
    end
end
