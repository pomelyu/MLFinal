function model = trainGaussianSVM( train_label, train_inst, sigma, C )

n = size(sigma,2);
m = size(C,2);

E = zeros(n,m);

for i=1:n
    for j = 1:m
        paraStr = ['-s 0 -t 2 -g ' num2str(0.5/power(sigma(1,i),2)) ... 
                   ' -c ' num2str(C(1,j)) ' -v 5 -q'];
        E(i,j) = svmtrain(train_label, train_inst, paraStr);
    end
end

[E, tmp] = max(E);
[E, j] = max(E);
i = tmp(1, j);

paraStr = ['-s 0 -t 2 -g ' num2str(0.5/power(sigma(1,i),2)) ... 
           ' -c ' num2str(C(1,j)) ' -q'];
model = svmtrain(train_label, train_inst, paraStr);

end

