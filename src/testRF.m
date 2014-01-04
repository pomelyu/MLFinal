function [predict_label, Err] = testRF( test_label, test_inst, model )

addpath('./lib/RFlib');

predict_label = forestTest(model, test_inst);

Err = sum(test_label ~= predict_label)/size(test_label, 1);

rmpath('./lib/RFlib');

end

