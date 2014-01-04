function model = trainAdaboost( train_label, train_inst, it)

addpath('./lib/GML_AdaBoost_Matlab_Toolbox');

model = cell(12,12);

% divide data to 12 catalog
DataSet  = cell(1, 12);
DataSize = zeros(1, 12);
for i = 1:12
    tmp = (train_label == i);
    DataSet{1, i} = train_inst(tmp, :);
    DataSize(1, i) = size(find(tmp > 0), 1);
end

% perform OVO
%   for example label 2 vs label 1
%   1 for label 2, -1 for label 1
% rStr = '';
k = 0;
weak_learner = tree_node_w(3);
for i = 1:12
    for j = 1:i-1
        tic
        [RLearners, RWeights] = GentleAdaBoost(weak_learner, ...
            [DataSet{1,i}; DataSet{1,j}]', ...
            [ones(DataSize(1,i),1); -ones(DataSize(1,j),1)]', it);
        model{i,j} = {RLearners, RWeights};
        toc
        %% Reveal progress
        k = k+1;
        fprintf('-- Done %02d/66\n', k);
%         msg = sprintf('-- Done %02d/66', k);
%         fprintf([rStr msg]);
%         rStr = repmat(sprintf('\b'),1,length(msg));
    end
end

fprintf('\n');

rmpath('./lib/GML_AdaBoost_Matlab_Toolbox');

end

