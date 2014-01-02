function model = trainAdaboost( train_inst, train_label )

options.weaklearner  = 0;
model = gentleboost_model(train_inst, train_label, options);

end

