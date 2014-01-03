function model = trainAdaboost( train_label, train_inst)

rmpath('./lib/libsvm');
addpath('./lib/gentleboost');

options.weaklearner  = 0;
model = gentleboost_model(train_inst, train_label, options);

addpath('./lib/libsvm');
rmpath('./lib/gentleboost');

end

