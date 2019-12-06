function res = Bessel_num_int(f, u, t, w)
%Computes the numerical integration after multiplying by Bessel function
% Please refer to Eq. (15) for better understanding the parameters
% param f(t): the feature
% param u: sample points, discretizing the final result
% param t: discrete sampling points of f
% param w: the weights used in approximating the integration
% return res: the final result

jb = besselj(0, u * t.');
res = u .* (jb * bsxfun(@times, t .* w, f));

end

