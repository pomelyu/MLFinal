function Eout = testLinearSVM(test_label, test_inst, model)

addpath('../lib/libsvm');

[predict_label, Eout, prob_estimates] = svmpredict(test_label, test_inst, model);

end