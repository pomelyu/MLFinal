function RunAdaBoost

addpath('./lib/gentleboost')
load('./data/train_crop.mat')
DataSet  = cell(1, 12);
DataSize = zeros(1, 12);

for i = 1:12
    tmp = (train_label == i);
    DataSet{1, i} = train_inst(tmp, :);
    DataSize(1, i) = size(find(tmp > 0), 1);
end

model = cell(12,12); 
options.weaklearner = 0;
options.T           = 10;

for i = 6:12
    for j = 1:i-1
        load('./save/model_Adaboost_crop_crop.mat');
        tic
        tmp = gentleboost_model([DataSet{1,i}; DataSet{1,j}], [ones(DataSize(1,i),1); -ones(DataSize(1,j),1)], options);
        model{i,j}  = tmp;
        toc
        save('./save/model_Adaboost_crop_crop.mat', 'model');
    end
end

end

