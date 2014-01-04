function model = trainRF( train_label, train_inst )

addpath('./lib/RFlib');

options.depth           = 10;
options.numTree         = 100;
options.numSplits       = 2;
options.classifierID    = 2;

model = forestTrain(train_inst, train_label, options);

rmpath('./lib/RFlib');

end

