function model = trainLinearSVM( train_label, train_inst )

addpath('./lib/libsvm');

C = 1;

paraStr = ['-s 0 -t 0 -c ' num2str(C) ' -q'];
model = svmtrain(train_label, train_inst, paraStr);

end

