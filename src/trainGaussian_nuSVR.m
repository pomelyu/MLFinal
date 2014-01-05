function model = trainGaussian_nuSVR( train_label, train_inst )

Sigma = power(2, [-2 0 2]);
Gamma = (2 .* Sigma .* Sigma) .^ (-1);
C = power(2, [-3 -2 -1 0 1]);

sizeG = size(Gamma, 2);
sizeC = size(C, 2);

E = zeros(sizeG, sizeC);

for i = 1:sizeG
    for j = 1:sizeC
        paraStr = ['-s 4 -t 2 -h 0 -g ', num2str(Gamma(1,i)),...
            ' -c ', num2str(C(1,j)), ' -n 0.5 -v 5 -q'];
        E(i,j) = svmtrain(train_label(1:3000,:), train_inst(1:3000,:), paraStr);
    end
end

[E, i] = min(E);
[~, j] = min(E);

i = i(1,j);

paraStr = ['-s 4 -t 2 -h 0 -g ', num2str(Gamma(1,i)),...
    ' -c ', num2str(C(1,j)), ' -n 0.5 -q'];

% Sigma = power(2, 2);
% Gamma = (2 .* Sigma .* Sigma) .^ (-1);
% C = power(2, 1);
% 
% paraStr = ['-s 4 -t 2 -h 0 -g ', num2str(Gamma), ' -c ', num2str(C), ' -n 0.5 -q'];

model = svmtrain(train_label, train_inst, paraStr);

end

