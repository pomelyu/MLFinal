function downSample_inst = DownSampling( data_inst )

% base on the training data is 122x105 image
n = size(data_inst,1);
tmp = zeros(n, 73*61);

for dataIdx = 1:n
    k = 1;
    for i = 1:2:105
        for j = 1:2:122
            tmp(dataIdx,k) = data_inst(dataIdx,j+122*(i-1));
            k = k+1;
        end
    end
end

downSample_inst = sparse(tmp);

end

