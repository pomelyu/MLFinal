function MLFinal()

% directory of all the file we write 
addpath('./src');
% directory to save temporary/final result 
addpath('./result');

fid = fopen('./result/result.txt', 'w');

trainData = 0;
testData = 0;

%% implement
while 1
    fprintf('-- Choose the number of Problem --\n');
    fprintf('   [1] Linear SVM.\n');
    fprintf('-- ---------------------------- --\n');
    fprintf('   [R] Read training data\n');
    fprintf('   [T] Read test data\n');
    fprintf('   [P] Perform prediction on test data\n')
    fprintf('   [E] Exit.\n');
    
    type = input('-- Type = ? ', 's');
    
    switch type
        case '1'
            %% TODO
        
        % Read training data
        case 'R'
            file_name = input('-- Enter the training data or remain blank for default -- ', 's');
            trainData = ReadData(file_name, 'ml2013final_train.dat');
            
        % Read test data
        case 'T'
            file_name = input('-- Enter the testing data or remain blank for default -- ', 's');
            testData = ReadData(file_name, 'ml2013final_test1.nolabel.dat');
            
        % Perform prediction on test data
        case 'P'
            %% TODO
            
        % Exit    
        case 'E'
            fprintf('-- Exit\n');
            break;
            
        % Undefine command    
        otherwise
            fprintf('-- Err, type undefined!\n');
    end
end

fclose(fid);

end


function Data = ReadData(file_name, default_name)
    % Data is an 1xN cell.
    % Data(1,i) is an data(D), which is Nx2 array
    % D(1,1) is the label of the D, D(1,2) is always zero
    % D(i,1) and D(i,2) is the pixel of character and the pixel value serparately 


    % if file_name is empty, use default_name
    if strcmp(file_name, '')
        file_name = default_name;
    end
    
    file_id = fopen(file_name, 'r');
    oneLine = fgets(file_id);
    Data = cell(1,1);
    Data{1,1} = Line2Data(oneLine);
    
    while ischar(oneLine)
        oneLine = fgets(file_id);
        Data = { Data{:,:} Line2Data(oneLine)};
    end
    
    %% convert one line to data
    function data = Line2Data(line)
        lineCell = strsplit(line, {' ', ':'});
        L = size(lineCell, 2);
        data = zeros(1,L+1);
        data(1,1) = str2double(lineCell{1,1});
        for i = 2:L
            data(1,i+1) = str2double(lineCell{1,i});
        end
        data = reshape(data, 2, (L+1)/2)';
    end

end

