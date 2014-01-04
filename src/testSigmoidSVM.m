function [predict_label, Err] = testSigmoidSVM( test_label, test_inst, model )

[predict_label, accuracy, ~] = svmpredict(test_label, test_inst, model);
Err = 1 - accuracy(1,1)/100;

end

