function Err = testLinearSVM(test_label, test_inst, model)

%addpath('./lib/libsvm');

[predict_label, accuracy, prob_estimates] = svmpredict(test_label, test_inst, model);
Err = 1 - accuracy(1,1)/100;

end