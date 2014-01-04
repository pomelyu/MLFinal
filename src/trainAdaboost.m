function model = trainAdaboost( train_label, train_inst)

rmpath('./lib/libsvm');
addpath('./lib/gentleboost');

% divide data to 12 catalog
DataSet  = cell(1, 12);
DataSize = zeros(1, 12);
for i = 1:12
    tmp = (train_label == i);
    DataSet{1, i} = train_inst(tmp, :);
    DataSize(1, i) = size(find(tmp > 0), 1);
end

model = cell(12,12);

% perform OVO
%   for example label 2 vs label 1
%   1 for label 2, -1 for label 1
rStr = '';
k = 0;
options.weaklearner = 0;
options.T           = 10;
for i = 1:12
    for j = 1:i-1
        model{i,j} = gentleboost_model([DataSet{1,i}; DataSet{1,j}], [ones(DataSize(1,i),1); -ones(DataSize(1,j),1)], options);
        
        %% Reveal progress
        k = k+1;
        msg = sprintf('-- Done %02d/66', k);
        fprintf([rStr msg]);
        rStr = repmat(sprintf('\b'),1,length(msg));
    end
end
fprintf('\n');

% options.weaklearner = 0;
% options.T           = 10;
% model = gentleboost_model(train_inst, train_label, options);

rmpath('./lib/gentleboost');
addpath('./lib/libsvm');

end

