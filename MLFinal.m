function MLFinal()

% directory of all the file we write 
addpath('./src');
% directory to save temporary data 
addpath('./save');
% directory of libsvm
addpath('./lib/libsvm');

%% training mode
model_name  = 'none';
model_idx   = 0;

%% training data
train_name  = 'raw';
train_idx   = 0;
valid_name  = 'raw';
valid_idx   = 0;

%% test data
%test_name   = 'raw';
test_idx    = 0;


%% implement
while 1
    % ========== Add training model chice here ==========
    fprintf('==================================\n');
    fprintf('-- Choose the number of Problem --\n');
    fprintf('   [1] Linear SVM.\n');
    fprintf('   [3] Deep Belief Network (DBN)\n')
    fprintf('   [4] Gaussian Kernel SVM\n');
    fprintf('   [7] Multi-class Adaboost\n');
    fprintf('  [10] Random Forest\n');
    fprintf('  [13] Polynomial Kernel SVM\n');
    fprintf('  [16] Sigmoid Kernel SVM\n');
    fprintf('----------------------------------\n');
    % ========== End Add model choice ===================
    
    fprintf('   [R] Read raw training data\n');
    fprintf('   [T] Read raw test data\n');
    fprintf('   [P] Prediction with model %s\n', model_name);
    fprintf('   [C] Calculate Ein with model %s\n', model_name);
    fprintf('   [B] Blending by voting\n')
    fprintf('   [E] Exit.\n');
    fprintf('==================================\n');   
    
    type = input('-- Type = ? ', 's');
    
    switch type
        % ========== Add training model here ================
        % ==== LinearSVM ====
        case '1'
            model_name = 'LinearSVM';
            model_idx  = 1;
            [valid_inst, train_inst, train_label] = ChooseTrainData();
            op = ['./save/model_' model_name '_' valid_name '_' train_name '.mat'];
            % if model already exist, just load to workspace
            if exist(op, 'file') == 2
                load(op);
            else
                C = [1 0.1 0.01 0.001 0.0001];
                model = trainLinearSVM(valid_inst, train_label, train_inst, C);
                save(op, 'model');
            end
            clear valid_inst train_inst train_label;
            
        % ==== DBN ====    
        case '3'
            model_name = 'DBN';
            model_idx  = 3;
            [~, train_inst, train_label] = ChooseTrainData();
            op = ['./save/model_' model_name '_' valid_name '_' train_name '.mat'];
            % if model already exist, just load to workspace
            if exist(op, 'file') == 2
                load(op);
            else
                model = trainDBN(train_label, train_inst);
                save(op, 'model');
            end
            clear valid_inst train_inst train_label;
            
        % ==== Gaussian SVM ====    
        case '4'
            model_name = 'GaussianSVM';
            model_idx  = 4;
            [valid_inst, train_inst, train_label] = ChooseTrainData();
            op = ['./save/model_' model_name '_' valid_name '_' train_name '.mat'];
            % if model already exist, just load to workspace
            if exist(op, 'file') == 2
                load(op);
            else
                sigma = [10 100 1000];
                C = [0.1 1 10];
                model = trainGaussianSVM(valid_inst, train_label, train_inst, sigma, C);
                save(op, 'model');
            end
            clear valid_inst train_inst train_label;            
        % ==== Adaboost ====
        case '7'
            model_name = 'Adaboost';
            model_idx  = 7;
            [~, train_inst, train_label] = ChooseTrainData();
            op = ['./save/model_' model_name '_' valid_name '_' train_name '.mat'];
            % if model already exist, just load to workspace
            if exist(op, 'file') == 2
                load(op);
            else
                model = trainAdaboost(train_label(:,:), train_inst(:,:));
                save(op, 'model');
            end
            clear valid_inst train_inst train_label;
        % ==== Random Forest ====
        case '10'
            model_name = 'RF';
            model_idx  = 10;
            [~, train_inst, train_label] = ChooseTrainData();
            op = ['./save/model_' model_name '_' valid_name '_' train_name '.mat'];
            % if model already exist, just load to workspace
            if exist(op, 'file') == 2
                load(op);
            else
                model = trainRF(train_label(:,:), train_inst(:,:));
                save(op, 'model');
            end
            clear valid_inst train_inst train_label;
            
        % ==== Polynomial SVM ====
        case '13'
            model_name = 'PolySVM';
            model_idx  = 13;
            [~, train_inst, train_label] = ChooseTrainData();
            op = ['./save/model_' model_name '_' valid_name '_' train_name '.mat'];
            % if model already exist, just load to workspace
            if exist(op, 'file') == 2
                load(op);
            else
                model = trainPolySVM(train_label(:,:), train_inst(:,:));
                save(op, 'model');
            end
            clear valid_inst train_inst train_label;
            
        case '16'
            model_name = 'SigmoidSVM';
            model_idx  = 16;
            [~, train_inst, train_label] = ChooseTrainData();
            op = ['./save/model_' model_name '_' valid_name '_' train_name '.mat'];
            % if model already exist, just load to workspace
            if exist(op, 'file') == 2
                load(op);
            else
                model = trainSigmoidSVM(train_label(:,:), train_inst(:,:));
                save(op, 'model');
            end
            clear valid_inst train_inst train_label;
            
        % ========== End training model ====================
        
        
        
        
        % Read training data
        case 'R'
            file_name = input('-- Enter the training data or remain blank for default -- ', 's');
            if strcmp(file_name, '')
                file_name = 'ml2013final_train.dat';
            end
            [train_label, train_inst] = libsvmread(file_name);
            save ./data/train_raw.mat train_label train_inst;
            
            
        % Read test data
        case 'T'
            file_name = input('-- Enter the test data or remain blank for default -- ', 's');
            if strcmp(file_name, '')
                if exist('./dat/test_raw.mat', 'file') == 2
                    load ./data/test_raw.mat
                else
                    [test_label, test_inst] = libsvmread('ml2013final_test1.nolabel.dat');
                    save ./data/test_raw.mat test_label test_inst;
                end
            else
                [test_label, test_inst] = libsvmread(file_name);
                save ./data/test_raw.mat test_label test_inst;
            end
            
        % Calculate Ein with model
        case 'C'
            [test_inst, test_label] = ChooseTestData('train');
            [~, Ein] = TestModel(test_label, test_inst, model, model_idx);
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
            fprintf(fid, ['Perform model with model ' model_name '_' valid_name '_' train_name '\n']);
            fprintf(fid, '-- Done with Ein = %2.2f%%\n\n', Ein*100);
            fclose(fid);
            clear predict_label test_label test_inst model;
            
        % Perform prediction on test data
        case 'P'
            [test_inst, test_label] = ChooseTestData('test');
            [predict_label, Eout] = TestModel(test_label, test_inst, model, model_idx);
            fprintf('-- Done with Eout = %2.2d%\n\n', Eout*100);
            
            % print predict label in ./result/model_x_predict.txt
            fid = fopen(['./result/model_' model_name '_' valid_name '_' train_name '_predict.txt'], 'w');
            for i=1:size(predict_label,1);
                fprintf(fid, '%d\n', predict_label(i,1));
            end
            fclose(fid);
            
            clear predict_label test_label test_inst model;
        
        % Perform blending on file in ./result
        case 'B'
            predict_label = Blending();           
            fid = fopen('./result/blending.txt', 'w');
            for i=1:size(predict_label,1);
                fprintf(fid, '%d\n', predict_label(i,1));
            end
            fclose(fid);
            
        % Exit    
        case 'E'
            fprintf('-- Exit\n');
            break;
            
        % Undefine command    
        otherwise
            fprintf('-- Err, type undefined!\n');
    end
end


    %% ChooseTrainData
    function [valid_inst, train_inst, train_label]= ChooseTrainData()
        %% Validation data
        fprintf('-- Choose the validation data --\n');
        fprintf('   [1] Raw image.\n');
        fprintf('   [2] Image with downsampling\n');
        fprintf('   [3] Image with cropping\n');
        fprintf('   [4] Image with cropping and binarization\n');
        fprintf('---------------------------------\n');
        valid_idx = input('-- Type ? ');
        
        % load raw data
        if exist('./data/train_raw.mat', 'file') == 2
            load ./data/train_raw.mat
        else
            fprintf('-- Loadind raw training data\n');
            [train_label, train_inst] = libsvmread('ml2013final_train.dat');
            save ./data/train_raw.mat train_label train_inst;
        end
        
        switch valid_idx   
            case 2
                valid_name  = 'down';
                if exist('./data/train_downsample.mat', 'file') == 2
                    load ./data/train_downsample.mat
                else
                    fprintf('-- Perform downsampling\n');
                    train_inst = DownSampling(train_inst);
                    save ./data/train_downsample train_label train_inst;
                end
                
            case 3
                valid_name  = 'crop';
                if exist('./data/train_crop.mat', 'file') == 2
                    load ./data/train_crop.mat
                else
                    fprintf('-- Perform cropping\n');
                    train_inst = ImgCropping(train_inst);
                    save ./data/train_crop train_label train_inst;
                end
            case 4
                valid_name  = 'crop_binary';
                if exist('./data/train_crop_binary.mat', 'file') == 2
                    load ./data/train_crop_binary.mat
                else
                    load ./data/train_crop.mat
                    train_inst = Binarize(train_inst);
                    save ./data/train_crop_binary.mat train_label train_inst;
                end
            otherwise
                valid_name  = 'raw';
        end
        
        valid_inst = train_inst;
                
        %% Training data
        fprintf('-- Choose the training data --\n');
        fprintf('   [1] Raw image.\n');
        fprintf('   [2] Image with downsampling\n');
        fprintf('   [3] Image with cropping\n');
        fprintf('   [4] Image with cropping and binarization\n');
        fprintf('---------------------------------\n');
        train_idx = input('-- Type ? ');
        
        % load raw data
        if exist('./data/train_raw.mat', 'file') == 2
            load ./data/train_raw.mat
        else
            fprintf('-- Loadind raw training data\n');
            [train_label, train_inst] = libsvmread('ml2013final_train.dat');
            save ./data/train_raw.mat train_label train_inst;
        end
        
        switch train_idx   
            case 2
                train_name  = 'down';
                if exist('./data/train_downsample.mat', 'file') == 2
                    load ./data/train_downsample.mat
                else
                    fprintf('-- Perform downsampling\n');
                    train_inst = DownSampling(train_inst);
                    save ./data/train_downsample train_label train_inst;
                end
                
            case 3
                train_name  = 'crop';
                if exist('./data/train_crop.mat', 'file') == 2
                    load ./data/train_crop.mat
                else
                    fprintf('-- Perform cropping\n');
                    train_inst = ImgCropping(train_inst);
                    save ./data/train_crop train_label train_inst;
                end
            case 4
                train_name  = 'crop_binary';
                if exist('./data/train_crop_binary.mat', 'file') == 2
                    load ./data/train_crop_binary.mat
                else
                    load ./data/train_crop.mat
                    train_inst = Binarize(train_inst);
                    save ./data/train_crop_binary.mat train_label train_inst;
                end
            otherwise
                train_name  = 'raw';
        end
    end

    %% ChooseTestData
    function [test_inst, test_label] = ChooseTestData(op)
        %% Validation data
        fprintf('-- Choose the test data --\n');
        fprintf('   [1] Raw image.\n');
        fprintf('   [2] Image with downsampling\n');
        fprintf('   [3] Image with cropping\n');
        fprintf('   [4] Image with cropping and binarization\n');
        fprintf('---------------------------------\n');
        test_idx = input('-- Type ? ');
        
        if strcmp(op, 'test')
            if exist('./data/test_raw.mat', 'file') == 2
                load ./data/test_raw.mat
            else
                fprintf('-- Loadind raw test data\n');
                [test_label, test_inst] = libsvmread('ml2013final_test1.nolabel.dat');
                save ./data/test_raw.mat test_label test_inst;
            end
            
            switch test_idx
                case 2
                    %test_name = 'down';
                    if exist('./data/test_downsampling.mat', 'file') == 2
                        load ./data/test_downsampling.mat
                    else
                        fprintf('-- Perform downsampling\n');
                        test_inst = DownSampling(test_inst);
                        save ./data/test_downsampling.mat test_label test_inst;
                    end
                    
                case 3
                    %test_name = 'crop';
                    if exist('./data/test_crop.mat', 'file') == 2
                        load ./data/test_crop.mat
                    else
                        fprintf('-- Perform cropping\n');
                        test_inst = ImgCropping(test_inst);
                        save ./data/test_crop test_label test_inst;
                    end
                case 4
                    %test_name  = 'crop_binary';
                    if exist('./data/test_crop_binary.mat', 'file') == 2
                        load ./data/test_crop_binary.mat
                    else
                        load ./data/test_crop.mat
                        test_inst = Binarize(test_inst);
                        save ./data/test_crop_binary.mat;
                    end
                otherwise
                    %test_name = 'raw';
            end
            
        else
            S = load('./data/train_raw.mat');
            switch test_idx
                case 2
                    %test_name = 'down';
                    P = load('./data/train_downsampling.mat');
                    test_label = S.train_label;
                    test_inst  = P.train_inst;
                    
                case 3
                    %test_name = 'crop';
                    P = load('./data/train_crop.mat');
                    test_label = S.train_label;
                    test_inst  = P.train_inst;
                case 4
                    P = load('./data/train_crop_binary.mat');
                    test_label = S.train_label;
                    test_inst  = P.train_inst;
                    
                otherwise
                    %test_name = 'raw';
                    test_label = S.train_label;
                    test_inst  = S.train_inst;
            end
        end
        
    end

end

function [predict_label, Eout] = TestModel(test_label, test_inst, model, model_idx)

switch(model_idx);
    % ========== Add model testing here ================
    case 1
        [predict_label, Eout] = testLinearSVM(test_label, test_inst, model);
    case 3
        [predict_label, Eout] = testDBN(test_label, test_inst, model);
    case 4
        [predict_label, Eout] = testGaussianSVM(test_label, test_inst, model);
    case 7
        [predict_label, Eout] = testAdaboost(test_label, test_inst, model);
    case 10
        [predict_label, Eout] = testRF(test_label, test_inst, model);
    case 13
        [predict_label, Eout] = testPolySVM(test_label, test_inst, model);
    case 16
        [predict_label, Eout] = testSigmoidSVM(test_label, test_inst, model);
    % ========== End model testing =====================
    otherwise
        fprintf('-- Please training data first\n');
end

end
