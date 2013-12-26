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
    fprintf('   [4] Linear SVM with downSampling\n');
    fprintf('----------------------------------\n');
    fprintf('   [R] Read training data\n');
    fprintf('   [T] Read test data\n');
    fprintf('   [P] Prediction with model %s\n', model_name);
    fprintf('   [C] Calculate Ein with model %s\n', model_name);
    fprintf('   [E] Exit.\n');
    fprintf('==================================\n');
    
    % read default training data
    if exist('./save/trainData.mat', 'file') == 2
        load ./save/trainData.mat
    else
        fprintf('-- Loadind default training data\n');
        [train_raw_label, train_raw_inst] = libsvmread('ml2013final_train.dat');
        save ./save/trainData.mat train_raw_label train_raw_inst;
    end
    
    type = input('-- Type = ? ', 's');
    
    switch type
        % ========== Add training model here ================
        case '1'
            % if model already exist, just load to workspace
            if exist('./save/model_LinearSVM.mat', 'file') == 2
                load ./save/model_LinearSVM.mat
                model_name = 'Linear SVM';
                model_idx = 1;
            else
                if exist('train_raw_inst', 'var') == 1 
                    C = [1 0.1 0.01 0.001 0.0001];
                    model = trainLinearSVM(train_raw_label, train_raw_inst, C);
                    model_name = 'Linear SVM';
                    model_idx = 1;
                    save ./save/model_LinearSVM.mat model
                else
                    fprintf('-- Please read training data\n')
                end
            end
            
        case '4'
            % if model already exist, just load to workspace
            if exist('./save/model_LinearSVM_DownSampling.mat', 'file') == 2
                load ./save/model_LinearSVM_DownSampling.mat
                model_name = 'Linear SVM with DownSampling';
                model_idx = 4;
            else
                if exist('train_raw_inst', 'var') == 1
                    if exist('./save/train_downsampling_inst.mat', 'file') == 2
                        load ./save/train_downsampling_inst.mat
                    else
                        train_downsampling_inst = DownSampling(train_raw_inst);
                        save ./save/train_downsampling_inst.mat train_downsampling_inst
                    end
                    fprintf('-- End downSampling\n');
                    C = 0.008:0.002:0.016;
                    model = trainLinearSVM(train_raw_label, train_downsampling_inst, C);
                    model_name = 'Linear SVM with DownSampling';
                    model_idx = 4;
                    save ./save/model_LinearSVM_DownSampling.mat model
                else
                    fprintf('-- Please read training data\n');
                end
            end
            
        % ========== End training model ====================
        % Read training data
        case 'R'
            file_name = input('-- Enter the training data or remain blank for default -- ', 's');
            if strcmp(file_name, '')
                file_name = 'ml2013final_train.dat';
            end
            [train_raw_label, train_raw_inst] = libsvmread(file_name);
            save ./save/trainData.mat train_raw_label train_raw_inst;
            
            
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
            
        % Calculate Ein with model
        case 'C'
            if exist('train_raw_inst', 'var') == 1
               Ein = TestModel(train_raw_label, train_raw_inst, model, model_idx);
               fprintf('-- Done with Ein = %2.2f%%\n', Ein*100);
            else
               fprintf('-- Please read train data\n');
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
    % ========== Add model testing here ================
    case 1
        Eout = testLinearSVM(test_label, test_inst, model);
    case 4
        Eout = testLinearSVM(test_label, test_inst, model);
    % ========== End model testing =====================
    otherwise
        printf('-- Please training data first\n');
end

end
