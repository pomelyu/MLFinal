function MLFinal()

% directory of all the file we write 
addpath('./src');
% directory to save temporary data 
addpath('./save');
% directory of libsvm
addpath('./lib/libsvm');

fid = fopen('log.txt', 'w');

% training mode
model_name = 'None';
model_idx  = 0;

%% implement
while 1
    fprintf('==================================\n');
    fprintf('-- Choose the number of Problem --\n');
    fprintf('   [1] Linear SVM.\n');
    fprintf('----------------------------------\n');
    fprintf('   [R] Read training data\n');
    fprintf('   [T] Read test data\n');
    fprintf('   [P] Prediction with model %s\n', model_name);
    fprintf('   [E] Exit.\n');
        fprintf('==================================\n');
    
    type = input('-- Type = ? ', 's');
    
    switch type
        case '1'
            % if model already exist, just load to workspace
            if exist('./save/model_LinearSVM.mat', 'file') == 2
                load ./save/model_LinearSVM.mat
                model_name = 'Linear SVM';
                model_idx = 1;
            else
                C = 1;
                if exist('train_raw_inst', 'var') == 1
                    paraStr = ['-s 0 -t 0 -c ' num2str(C) ' -q'];
                    model = svmtrain(train_raw_label, train_raw_inst, paraStr);
                    model_name = 'Linear SVM';
                    model_idx = 1;
                    save ./save/model_LinearSVM.mat model
                else
                    fprintf('-- Please read training data\n')
                end
            end
        
        % Read training data
        case 'R'
            file_name = input('-- Enter the training data or remain blank for default -- ', 's');
            if strcmp(file_name, '')
                if exist('./save/trainData.mat', 'file') == 2
                    load ./save/trainData.mat
                else
                    [train_raw_label, train_raw_inst] = libsvmread('ml2013final_train.dat');
                    save ./save/trainData.mat train_raw_label train_raw_inst;
                end
            else
                [train_raw_label, train_raw_inst] = libsvmread(file_name);
                save ./save/trainData.mat train_raw_label train_raw_inst;
            end
            
            
        % Read test data
        case 'T'
            file_name = input('-- Enter the test data or remain blank for default -- ', 's');
            if strcmp(file_name, '')
                if exist('./save/testData.mat', 'file') == 2
                    load ./save/testData.mat
                else
                    [test_raw_label, test_raw_inst] = libsvmread('ml2013final_test1.nolabel.dat');
                    save ./save/testData.mat test_raw_label test_raw_inst;
                end
            else
                [test_raw_label, test_raw_inst] = libsvmread(file_name);
                save ./save/testData.mat test_raw_label test_raw_inst;
            end
            
            
        % Perform prediction on test data
        case 'P'
            if exist('test_raw_inst', 'var') == 1
               Eout = TestModel(test_raw_label, test_raw_inst, model, model_idx);
               fprintf('-- Done with Eout = %2.2d%\n', Eout*100);
            else
               fprintf('-- Please read test data\n');
            end
           
            
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

function Eout = TestModel(test_label, test_inst, model, model_idx)

switch(model_idx);
    case 1
        [predict_label, Eout, prob_estimates] = svmpredict(test_label, test_inst, model);
    otherwise
        printf('-- Please training data first\n');
end

end
