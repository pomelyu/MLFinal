function [newcenter, sigma, W, yh, rmse] = ML2013Final_RBF_OLS_Train(p, t, tol)
% p is the input data, can be a N X K matrix, N is the dimension of teach
% data, K is the no. of the data
% t is the target value, is a 1 X K matrix
% nc is the no. of the predefined centers
% newcenter is the locations of the predefined centers, is a nc X N matrix
% sigma is the constant standard deviation
% W is the weights, is a nc X 1 matrix
% yh is the network output, is a 1 X K matrix
% rmse is the RMSE of the target and network output
% tol is the specified tolerance

p = p';
t = t';

[nd, np] = size(p);
nc = np; % nc=np denoting that in the beginning, each point is the center point
pp = p;
Dic = zeros(nc);

phi = zeros(np, nc);
s = zeros(np, nc);
SS = [];
err = zeros(nc, 1);
newcenter = [];
for i = 1:nc
    for j = 1:nc
        Dic(i, j) = norm(p(:, i) - p(:, j));
    end
end
sigma = max(max(Dic)) / sqrt(nc);
for i = 1:nc
    for j = 1:np % actually, now nc = np
        phi(j, i) = exp(-(norm(p(:, i) - p(:,j)) / sigma) ^ 2);
    end
end

k = 1;
tnorm = t * t'; % t is the target value
for i = 1:np % this loop calculate the error reducing ratio for each point
    s(:, i) = phi(:, i);
    h = s(:, i)' * s(:, i);
    g = s(:, i)' * t' / h;
    err(i) = h * g ^ 2 / tnorm;
end
[errmax, imax] = max(err); % imax is an index to indicate the location with max
newcenter = p(:, imax); % error reducing ratio
SS = phi(:, imax);
errsum = errmax;
p(:, imax) = []; % the point ever picked up will never be picked up again
phi(:, imax) = [];
err(imax) = [];
while(errsum < tol & k < np)
    k = k + 1;
    for i = 1:(np - k + 1)
        for j = 1:(k - 1)
            a(j, i) = SS(:, j)' * phi(:, i) / (SS(:, j)' * SS(:, j)); % the coeff. of the orthogonal vector (alpha)
        end
        s(:, i) = phi(:, i) - SS * a(:, i); % the orthogonal vector
        h = s(:, i)' * s(:, i);
        g = s(:, i)' * t' / h;
        err(i) = h * g ^ 2 / tnorm;
    end
    [errmax, imax] = max(err);
    errsum = errsum + errmax;
    newcenter = [newcenter p(:, imax)];
    SS = [SS s(:,imax)];
    p(:, imax) = [];
    phi(:, imax) = [];
    err(end) = [];
end
% finish the while loop, means that we have found the number and location
% of the center points
nc = size(newcenter, 2);
phi = zeros(np, nc);

% once we have known the no. and location of the centers, the rest codes is similar to the previous one
for i = 1:nc
    for j = 1:nc
        Dic(i,j) = norm(newcenter(:, i) - newcenter(:, j));
    end
end
sigma = max(max(Dic)) / sqrt(nc);
for i = 1:nc
    for j = 1:np
        phi(j, i) = exp(-(norm(newcenter(:, i) - pp(:, j)) / sigma) ^ 2);
    end
end
W = pinv(phi) * t';
yh = (phi * W)';
rmse = sqrt(mse(yh - t));
newcenter = newcenter';
