function [Y_Predict] = ML2013Final_RBF_KMeans_Predict(X_Test, newcenter, sigma, W)
N_Test = size(X_Test, 1);
nc = size(newcenter, 1);
for i = 1:N_Test
    Sum_Temp = 0;
    for j = 1:nc
        Sum_Temp = Sum_Temp + W(j) * exp(-(norm(X_Test(i, :) - newcenter(j, :)) / sigma) ^ 2);        
    end
    Y_Predict(i, 1) = sign(Sum_Temp);
end
