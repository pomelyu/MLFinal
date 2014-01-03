function [predict_label, Err] = testAdaboost( test_label, test_inst, model )

rmpath('./lib/libsvm');
addpath('./lib/gentleboost');

n = size(test_label, 1);

[predict_label , fxtrain] = gentleboost_predict(test_inst, model);
Err = sum(test_label ~= predict_label)/n;

addpath('./lib/libsvm');
rmpath('./lib/gentleboost');

end

