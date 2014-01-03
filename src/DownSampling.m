function downSample_inst = DownSampling( data_inst )

% base on the training data is width:105 height: 122 image
n = size(data_inst,1);
tmp = zeros(n, 53*61);

rStr = '';
for dataIdx = 1:n
    k = 1;
    for i = 1:2:122
        for j = 1:2:105
            tmp(dataIdx,k) = data_inst(dataIdx,j+105*(i-1));
            k = k+1;
        end
    end
    
    %% Reveal progress
    msg = sprintf('-- Done %04d/%04d', dataIdx, n);
    fprintf([rStr msg]);
    rStr = repmat(sprintf('\b'),1,length(msg));
end
fprintf('\n');

downSample_inst = sparse(tmp);

end

