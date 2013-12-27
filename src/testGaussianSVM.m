function Err = testGaussianSVM( test_label, test_inst, model )

[predict_label, accuracy, prob_estimates] = svmpredict(test_label, test_inst, model);
Err = 1 - accuracy(1,1)/100;

end