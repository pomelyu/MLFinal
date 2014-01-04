function model = trainPolySVM( train_label, train_inst )

% parameter
gamma = [0.01 0.1 1];
coef0 = [0.1 1 10];
degree = [2 3];

sizeG = size(gamma, 2);
sizeC = size(coef0, 2);
sizeD = size(degree, 2);

E = zeros(sizeG, sizeC, sizeD);

idx = randperm(size(train_label,1));
for i=1:sizeG
    for j = 1:sizeC
        for k = 1:sizeD
            paraStr = ['-s 0 -t 2 -d ' num2str(degree(1,k)) ...
                ' -g ' num2str(gamma(1,i)) ...
                ' -r ' num2str(coef0(1,j)) ' -v 5 -q'];
            E(i,j,k) = svmtrain(train_label(idx(1,1:2000)',:), ...
                train_inst(idx(1,1:2000)',:), paraStr);
        end
    end
end

[E, i] = max(E);
[E, j] = max(E);
[~, k] = max(E);

j = j(1,1,k);
i = i(1,j,k);

paraStr = ['-s 0 -t 2 -d ' num2str(degree(1,k)) ...
    ' -g ' num2str(gamma(1,i)) ...
    ' -r ' num2str(coef0(1,j)) ' -q'];

% paraStr = '-s 0 -t 2 -d 2 -g 0.01 -r 0.1 -q';

model = svmtrain(train_label, train_inst, paraStr);

end

