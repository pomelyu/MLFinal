function model = trainSigmoidSVM( train_label, train_inst )

% parameter
gamma = [0.01 0.1 1 10];
coef0 = [0.01 0.1 1 10];

sizeG = size(gamma, 2);
sizeC = size(coef0, 2);

E = zeros(sizeG, sizeC);

idx = randperm(size(train_label,1));
for i = 1:sizeG
    for j = 1:sizeC
        paraStr = ['-s 0 -t 3 -g ' num2str(gamma(1,i)) ...
            ' -r ' num2str(coef0(1,j)) ' -v 5 -q'];
        E(i,j) = svmtrain(train_label(idx(1,1:3000)',:), ...
            train_inst(idx(1,1:3000)',:), paraStr);
    end
end

[E, i] = max(E);
[~, j] = max(E);

i = i(1,j);

paraStr = ['-s 0 -t 3 -g ' num2str(gamma(1,i)) ...
    ' -r ' num2str(coef0(1,j)) ' -q'];

model = svmtrain(train_label, train_inst, paraStr);

end

