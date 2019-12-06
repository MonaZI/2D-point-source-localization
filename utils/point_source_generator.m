function X = point_source_generator(K, min_dist)

X = [0.5;0.5];
trial = 1;

% mask for extracting the distances


while length(X)<K
%     fprintf('trial = %d \n',trial)
    iter = 1;
    X = rand(2,1);
    while iter < 1000 && length(X)<K
        new_point = rand(2,1)-0.5;
        
        o_dist = sqrt(sum(X.^2,1));
        o_dist_tmp = abs(o_dist-norm(new_point));
        
        if length(find(o_dist_tmp<min_dist))==0 %&& (length(find(D_tmp<min_dist))==0 || length(X)==2)
            X = [X,new_point];
        end
        iter = iter + 1;
    end
    trial = trial + 1;
end

end