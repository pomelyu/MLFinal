function MLFinal()

% directory of all the file we write 
addpath('./src');
% directory to save temporary data 
addpath('./save');
% directory of libsvm
addpath('./lib/libsvm');

%% training mode
model_name = 'None';
model_idx  = 0;

% read default training data
if exist('./data/trainData.mat', 'file') == 2
    load ./data/trainData.mat
else
    fprintf('-- Loadind default training data\n');
    [train_raw_label, train_raw_inst] = libsvmread('ml2013final_train.dat');
    save ./data/trainData.mat train_raw_label train_raw_inst;
end


%% implement
while 1
    % ========== Add training model chice here ==========
    fprintf('==================================\n');
    fprintf('-- Choose the number of Problem --\n');
    fprintf('   [1] Linear SVM.\n');
    fprintf('   [4] Linear SVM with downSampling\n');
    fprintf('   [7] Gaussian Kernel SVM + downSampling\n');
    fprintf('   [10]Linear SVM with cropping\n');
    fprintf('   [13]Gaussian Kernel SVM + cropping\n');
    fprintf('----------------------------------\n');
    % ========== End Add model choice ===================
    fprintf('   [R] Read training data\n');
    fprintf('   [T] Read test data\n');
    fprintf('   [P] Prediction with model %s\n', model_name);
    fprintf('   [C] Calculate Ein with model %s\n', model_name);
    fprintf('   [E] Exit.\n');
    fprintf('==================================\n');   
    
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
                    model = trainLinearSVM(train_raw_inst, ...
                        train_raw_label, train_raw_inst, C);
                    model_name = 'Linear SVM';
                    model_idx = 1;
                    save ./save/model_LinearSVM.mat model
                else
                    fprintf('-- Please read training data\n')
                end
            end
        %    
        case '4'
            % if model already exist, just load to workspace
            if exist('./save/model_LinearSVM_DownSampling.mat', 'file') == 2
                load ./save/model_LinearSVM_DownSampling.mat
                model_name = 'Linear SVM with DownSampling';
                model_idx = 4;
            else
                if exist('train_raw_inst', 'var') == 1
                    if exist('./data/train_downsampling_inst.mat', 'file') == 2
                        load ./data/train_downsampling_inst.mat
                    else
                        train_downsampling_inst = DownSampling(train_raw_inst);
                        save ./data/train_downsampling_inst.mat train_downsampling_inst
                    end
                    fprintf('-- End downSampling\n');
                    C = 0.008:0.002:0.016;
                    model = trainLinearSVM(train_downsampling_inst,... 
                        train_raw_label, train_raw_inst, C);
                    model_name = 'Linear SVM with DownSampling';
                    model_idx = 4;
                    save ./save/model_LinearSVM_DownSampling.mat model
                else
                    fprintf('-- Please read training data\n');
                end
            end
            
        case '7'
            % if model already exist, just load to workspace
            if exist('./save/model_GaussianSVM_DownSampling.mat', 'file') == 2
                load ./save/model_GaussianSVM_DownSampling.mat
                model_name = 'Gaussian SVM with DownSampling';
                model_idx = 7;
            else
                if exist('train_raw_inst', 'var') == 1
                    if exist('./data/train_downsampling_inst.mat', 'file') == 2
                        load ./data/train_downsampling_inst.mat
                    else
                        train_downsampling_inst = DownSampling(train_raw_inst);
                        save ./data/train_downsampling_inst.mat train_downsampling_inst
                    end
                    fprintf('-- End downSampling\n');
                    sigma = [10 100 1000];
                    C = [0.1 1 10];
                    model = trainGaussianSVM(train_downsampling_inst, ...
                        train_raw_label, train_raw_inst, sigma, C);
                    model_name = 'Gaussian SVM with DownSampling';
                    model_idx = 7;
                    save ./save/model_GaussianSVM_DownSampling.mat model
                else
                    fprintf('-- Please read training data\n');
                end
            end

        case '10'
            % if model already exist, just load to workspace
            if exist('./save/model_LinearSVM_Cropping.mat', 'file') == 2
                load ./save/model_LinearSVM_Cropping.mat
                model_name = 'Linear SVM with Cropping';
                model_idx = 10;
            else
                if exist('train_raw_inst', 'var') == 1
                    if exist('./data/train_cropping_inst.mat', 'file') == 2
                        load ./data/train_cropping_inst.mat
                    else
                        train_cropping_inst = ImgCropping(train_raw_inst);
                        save ./data/train_cropping_inst.mat train_cropping_inst
                    end
                    fprintf('-- End Cropping\n');
                    C = [0.01 0.1 1 10];
                    model = trainLinearSVM(train_cropping_inst,... 
                        train_raw_label, train_cropping_inst, C);
                    model_name = 'Linear SVM with Cropping';
                    model_idx = 10;
                    save ./save/model_LinearSVM_Cropping.mat model
                else
                    fprintf('-- Please read training data\n');
                end
            end
            
        case '13'
            % if model already exist, just load to workspace
            if exist('./save/model_GaussianSVM_Cropping.mat', 'file') == 2
                load ./save/model_GaussianSVM_Cropping.mat
                model_name = 'Gaussian SVM with Cropping';
                model_idx = 13;
            else
                if exist('train_raw_inst', 'var') == 1
                    if exist('./data/train_cropping_inst.mat', 'file') == 2
                        load ./data/train_cropping_inst.mat
                    else
                        train_cropping_inst = DownSampling(train_raw_inst);
                        save ./data/train_cropping_inst.mat train_cropping_inst
                    end
                    fprintf('-- End Cropping\n');
                    sigma = [10 100 1000];
                    C = [0.1 1 10];
                    model = trainGaussianSVM(train_cropping_inst, ...
                        train_raw_label, train_cropping_inst, sigma, C);
                    model_name = 'Gaussian SVM with Cropping';
                    model_idx = 13;
                    save ./save/model_GaussianSVM_Cropping.mat model
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
            save ./data/trainData.mat train_raw_label train_raw_inst;
            
            
        % Read test data
        case 'T'
            file_name = input('-- Enter the test data or remain blank for default -- ', 's');
            if strcmp(file_name, '')
                if exist('./dat/testData.mat', 'file') == 2
                    load ./data/testData.mat
                else
                    [test_raw_label, test_raw_inst] = libsvmread('ml2013final_test1.nolabel.dat');
                    save ./data/testData.mat test_raw_label test_raw_inst;
                end
            else
                [test_raw_label, test_raw_inst] = libsvmread(file_name);
                save ./data/testData.mat test_raw_label test_raw_inst;
            end
            
        % Calculate Ein with model
        case 'C'
            if exist('train_raw_inst', 'var') == 1
               [predict_label, Ein] = TestModel(train_raw_label, train_raw_inst, model, model_idx);
               fprintf('-- Done with Ein = %2.2f%%\n', Ein*100);
               
               % If input 'y', record Ein in ./log/data_time.txt.
               % Otherwise, record Ein in ./log/tmp_log.txt
               isSave = input('-- Save the result ? [y]/[n]', 's');
               if strcmp(isSave, 'y')
                   file_name = [date ' ' num2str(hour(now)) ':' num2str(minute(now))];
               else
                   file_name = 'tmp_log';
               end
               fid = fopen(['./log/' file_name '.txt'], 'w');
               fprintf(fid, 'Perform model with model index %d\n', model_idx);
               fprintf(fid, '-- Done with Ein = %2.2f%%\n\n', Ein*100);
               fclose(fid);
            else
               fprintf('-- Please read train data\n');
            end
            
            
        % Perform prediction on test data
        case 'P'
            if exist('test_raw_inst', 'var') == 1
               [predict_label, Eout] = TestModel(test_raw_label, test_raw_inst, model, model_idx);
               fprintf('-- Done with Eout = %2.2d%\n', Eout*100);
               
               % print predict label in ./result/model_x_predict.txt
               fid = fopen(['./result/model_' num2str(model_idx) '_predict.txt'], 'w');
               for i=1:size(predict_label,1);
                   fprintf(fid, '%d\n', predict_label(i,1));
               end
               fclose(fid);
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

end

function [predict_label, Eout] = TestModel(test_label, test_inst, model, model_idx)

switch(model_idx);
    % ========== Add model testing here ================
    case 1
        [predict_label, Eout] = testLinearSVM(test_label, test_inst, model);
    case 4
        [predict_label, Eout] = testLinearSVM(test_label, test_inst, model);
    case 7
        [predict_label, Eout] = testGaussianSVM(test_label, test_inst, model);
    % ========== End model testing =====================
    otherwise
        fprintf('-- Please training data first\n');
end

end
