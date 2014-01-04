function model = trainAdaboost( train_label, train_inst)

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
it = 10;
rStr = '';
k = 0;
for i = 1:12
    for j = 1:i-1
        tic
        model{i,j} = AdaBoostOVO([ones(DataSize(1,i),1) ; -ones(DataSize(1,j),1)],...
            [DataSet{1,i} ; DataSet{1,j}], it);
        
        %% Reveal progress
        time = toc;
        k = k+1;
        msg = sprintf('-- Done %02d/66 in %.3f', k, time);
        fprintf([rStr msg]);
        rStr = repmat(sprintf('\b'),1,length(msg));
    end
end

fprintf('\n');

end

function model = AdaBoostOVO(label, inst, it)

model.Learner = cell(it,1);
model.Weight  = ones(it,1);

n = size(label, 1);
% initial u
u = ones(size(label,1), 1);
% perform it times
for k = 1:it
    [tmp_model, threshold] = weakLearner(label, inst, u);
    model.Learner{k,1} = tmp_model;
    
    % update u
    p_label = tmp_model.s*[-ones(threshold,1); ones(n-threshold,1)];
    %p_label = tmp_model.s * sign(inst(:, tmp_model.k) - tmp_model.theta);
    c_inst = (p_label == label);
    eps = sum((1 - c_inst) .* u)/sum(u);
    if eps == 0
        break;
    end
    u = u .* c_inst .* eps + u .* (1-c_inst) .* (1-eps);
    model.Weight = log(sqrt((1-eps)/eps));
end

end

function [model, threshold] = weakLearner(label, inst, u)

% dimension of feature
d = size(inst, 2);

model = [];
Err = intmax;
for k = 1:d
    inst_i = inst(:,k);
    [s, theta, E, threshold] = decisionStump(label, inst_i, u);
    if E < Err
        model.k = k;
        model.s = s;
        model.theta = theta;
        Err = E;
    end
    if Err == 0
        break;
    end
end

end

function [s, theta, Err, threshold] = decisionStump(label, inst, u)
% label: nx1 matrox
% inst:  nx1 matrix
% u:     nx1 matrix weight

% sort data in ascend order
[inst, idx] =sortrows(inst);
label = label(idx, 1);
u = u(idx, 1);

n = size(inst,1);

Err = intmax;
s = 0;
theta = 0;
% for positive ray
% 1~k is -1, otherwise is 1
for k = 1:n-1
    
    p_label = [-ones(k,1);ones(n-k,1)];
    E_pos = sum(     (p_label ~= label) .* u);
    E_neg = sum((1 - (p_label ~= label)).* u); 
    
%     for j = 1:k
%         i = idx(j,1);
%         if label(i,1) ~= -1
%             E_pos = E_pos + u(i,1);
%         else
%             E_neg = E_neg + u(i,1);
%         end
%     end
%     
%     for j = k:n
%         i = idx(j,1);
%         if label(i,1) ~= 1
%             E_pos = E_pos + u(i,1);
%         else
%             E_neg = E_neg + u(i,1);
%         end
%     end
% 

    if E_pos < Err
        Err = E_pos;
        s = 1;
        threshold = k;
        theta = (inst(k,1) + inst(k+1,1))/2;
    end
    if E_neg < Err
        Err = E_neg;
        s = -1;
        threshold = k;
        theta = (inst(k,1) + inst(k+1,1))/2;
    end
    if Err == 0
        break;
    end
end

end

