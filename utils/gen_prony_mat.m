function [prony_mat,prony_vec] = gen_prony_mat(b,K)
% generated the extended prony matrix and vector based on the samples b for
% K number of complex exponentials

M = length(b);

prony_mat = zeros(M-K,K);
prony_vec = zeros(M-K,1);
m = [M-2:-1:K-1];

for k = 1:length(m)
   prony_mat(m(k)-K+2,:) = b(m(k)+1:-1:m(k)-K+2);
   prony_vec(m(k)-K+2) = -b(m(k)+2);    
end



% generate the extended prony matrix
prony_mat1 = zeros(M-K,K);
prony_vec1 = zeros(M-K,1);
for k = 1:M-K
   prony_mat1(k,:) = conj(b(k+1:k+K));
   prony_vec1(k) = -conj(b(k));    
end

prony_mat = [prony_mat;prony_mat1];
prony_vec = [prony_vec;prony_vec1];

for k = 1:M-K
    prony_mat1(k,:) = b([k-1+K:-1:k]);
    prony_vec1(k) = b(k+1);
end

end