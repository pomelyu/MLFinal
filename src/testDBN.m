function [predict_label, Err]=testDBN(test_label, test_inst, model)

label_matrix = nnpredict(model, test_inst);
[predict_label, tmp] = find(label_matrix' > 0);

Err = sum(predict_label ~= test_label)/size(test_label, 1);

%assert(er < 0.10, 'Too big error');
