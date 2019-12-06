function plot_feature_baseline(f, r, locs, feature)
%Plots the estimated feature and compares it to the baseline
%:param f: the feature
%:param r: the radial shell
%:param locs: the baseline location of the peaks, i.e. radial or pairwise distances
%:param feature: the name of the feature

figure;
plot(r, f, 'LineWidth', 3)
hold on;
plot(locs, max(f) * ones(size(locs)), 'x', 'LineWidth', 3, 'MarkerSize', 8)
legend('Estimated', 'Groundtruth', 'Location', 'Best', 'FontSize', 14)
xlabel('$$u$$', 'FontSize', 14, 'interpreter', 'latex')
if strcmp(feature, 'Mean')
    ylabel('$$f_{\mu}(u)$$', 'interpreter', 'latex', 'FontSize', 14)
elseif strcmp(feature, 'Auto-correlation')
    ylabel('$$f_{C}(u)$$', 'interpreter', 'latex', 'FontSize', 14)
end

end

