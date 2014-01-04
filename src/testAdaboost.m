function [predict_label, Err] = testAdaboost( test_label, test_inst, model )

addpath('./lib/GML_AdaBoost_Matlab_Toolbox');

n = size(test_label, 1);

% Total 66 classifier C(12,2) = 12x11/2 = 66
predict_matrix = zeros(n, 66);

k = 0;
rStr = '';
for i = 1:12
    for j = 1:i-1
        k = k+1;
        %predict_class = adabost('apply', test_inst, model{i,j});
        S = model{i, j};
        predict_class = sign(Classify(S.RLearners, S.RWeights, test_inst'));
        predict_matrix(predict_class' ==  1, k) = i;
        predict_matrix(predict_class' == -1, k) = j;
        
        %% Reveal progress
        k = k+1;
        msg = sprintf('-- Done %02d/66', k);
        fprintf([rStr msg]);
        rStr = repmat(sprintf('\b'),1,length(msg));
    end
end
fprintf('\n');

predict_label = mode(perdict_matrix, 2);

Err = sum(test_label ~= predict_label)/n;

rmpath('./lib/GML_AdaBoost_Matlab_Toolbox');

end

