function model = trainLinearSVM( down_inst, train_label, train_inst, C )

%addpath('./lib/libsvm');

n = size(C,2);
E = zeros(1,n);

for i=1:n
    paraStr = ['-s 0 -t 0 -c ' num2str(C(1,i)) ' -v 5 -q'];
    E(1,i) = svmtrain(train_label, down_inst, paraStr);
end

[E, idx] = max(E);

paraStr = ['-s 0 -t 0 -c ' num2str(C(1,idx)) ' -q'];
model = svmtrain(train_label, train_inst, paraStr);

end

