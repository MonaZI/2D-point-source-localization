function coord = point_source_gen(numPoint, minDist, distType)
% generates the point-source model with #numPoint points with such that the
% minimum (either radial or pairwise) distance between the distances is minDist.
% param numPoint: the number of points
% param minDist: the minimum distance
% param distType: the distance for which we have the minimum distance
% criterion, choices are radial and pairwise
% return coord: the coordinates of the points

coord = [];
trial = 1;

while (length(coord) < numPoint)
    iter = 1;
    % the first point at the start of the trial
    coord = rand(2,1) - 0.5;
    o_dist = [norm(coord)];
    while (iter < 1000) && (length(coord) < numPoint)
        new_point = rand(2,1) - 0.5;
        if strcmp(distType, 'radial')
%             new_dist = sqrt(sum(new_point.^2,1));
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