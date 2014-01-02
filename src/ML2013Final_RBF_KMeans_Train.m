function [centers, sigma, W, yh, rmse] = ML2013Final_RBF_KMeans_Train(p, t, nc)
% p is the input data, can be a NxK matrix, N is the dimension of teach
% data, K is the no. of the data
% t is the target value, is a 1xK matrix
% nc is the no. of the predefined centers
% centers is the locations of the predefined centers, is a nc x N matrix
% sigma is the constant standard deviation
% W is the weights, is a ncx1 matrix
% yh is the network output, is a 1xK matrix
% rmse is the RMSE of the target and network output

p = p';
t = t';

[nd, np] = size(p);
% ---------------------------用 Matlab 內建 Kmeans 選取中心點
opts = statset('Display', 'final');
[idx, centers] = kmeans(p', nc, 'Distance', 'city', 'Replicates', 5, 'Options', opts);
centers= centers';
Dic = zeros(nc, nc);
phi = zeros(np, nc);
for i = 1:nc
    for j = 1:nc
        Dic(i, j) = norm(centers(:, i) - centers(:, j));
    end
end
sigma = max(max(Dic)) / sqrt(nc);
for i = 1:nc
    for j = 1:np
        phi(j, i) = exp(-(norm(centers(:, i) - p(:, j)) / sigma) ^ 2);
    end
end
W = pinv(phi) * t';
yh = (phi * W)';
rmse = sqrt(mse(yh - t));
centers = centers';
