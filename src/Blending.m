function predict_label = Blending()

outModel = dir('./result/model*.txt');
predict_label = [];

for i = 1:size(outModel,1)
    label = dlmread(['./result/' outModel(i,1).name]);
    predict_label = [predict_label label];
end

predict_label = mode(predict_label, 2);

end

