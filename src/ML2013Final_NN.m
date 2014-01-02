% ---------- 清除舊資料 ----------
clear all;
close all;
clc;
% ---------- 計時開始 ----------
tic;
% ---------- 讀取資料 ----------
[SVM_Y_Train, SVM_X_Train] = libsvmread('ml2013final_train.dat');
[SVM_Y_Test, SVM_X_Test] = libsvmread('ml2013final_test1.nolabel.dat');
SVM_X_Train = full(SVM_X_Train);
SVM_X_Test = full(SVM_X_Test);
SVM_X_Train_Zero_One = ceil(SVM_X_Train);
SVM_X_Test_Zero_One = ceil(SVM_Y_Test);
% ---------- 變數設定 ----------
[N_Train, Dimension_Train] = size(SVM_X_Train_Zero_One);
[N_Test, Dimension_Test] = size(SVM_X_Test_Zero_One);
[Total_Classes, ~] = max(SVM_Y_Train);

d = eye(2000, 2000);
pr = zeros(Dimension_Train, 2); 
pr(:,2) = ones(Dimension_Train , 1);
s = 2000;

net = newp(pr, s, 'logsig', 'learnwh'); % logsig is the activation function, learnwh is the LSM learning rule
net.inputweights{1, 1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net); %initialize the weights and the bias
% the parameter set up for training
net.trainParam.show = 500;
net.trainParam.epochs = 10; % number of iteration
net.trainParam.goal = 1e-4; % threshold value of converged
% network training
net = train(net, SVM_X_Train_Zero_One(1:2000, :)', d);
% obatining the weights and bias after training
wt = net.IW{1, 1};
bias = net.b{1};

% start to recognize this imperfect number using previous training results
for NT = 1 : (N_Train - 2000)
    % 預測
    Similarity_Predict = sim(net, SVM_X_Train_Zero_One(NT,:)');    
    
    % 最大值
    [Max_Similarity_Temp, Max_Index] = max(Similarity_Predict);
    Y_Predict(NT, 1) = SVM_Y_Train(Max_Index);
    
    % 隨機最大值
    Max_Random_Index = [];
    for i = 1:2000
        if Similarity_Predict(i, 1) == Max_Similarity_Temp
            Max_Random_Index = [Max_Random_Index; i];
        end
    end
    p2 = randperm(size(Max_Random_Index, 1));
    Max_Random_Index = Max_Random_Index(p2(1));
    Y_Predict(NT, 2) = SVM_Y_Train(Max_Random_Index);
    
    % 投票最大值    
    Need_Vote_Index = [];    
    for i = 1:2000
        if Similarity_Predict(i, 1) > 0.9
            Need_Vote_Index = [Need_Vote_Index; i];            
        end
    end    
    Class_Vote = zeros(1, Total_Classes);
    for i = 1:size(Need_Vote_Index, 1)
        Class = SVM_Y_Train(Need_Vote_Index(i));
        Class_Vote(1, Class) = Class_Vote(1, Class) + 1;
    end
    [Vote_Max, ~] = max(Class_Vote);
    Vote_Index = [];
    for KK = 1:Total_Classes
        if (Class_Vote(1, KK) == Vote_Max)
            Vote_Index = [Vote_Index, KK];
        end
    end        
    p3 =randperm(size(Vote_Index, 2));
    Y_Predict(NT, 3) = Vote_Index(p3(1));
end

A_1 = sum(Y_Predict(:, 1) == SVM_Y_Train(2000 +  1 : N_Train)) / (N_Train - 2000);
A_2 = sum(Y_Predict(:, 2) == SVM_Y_Train(2000 +  1 : N_Train)) / (N_Train - 2000);
A_3 = sum(Y_Predict(:, 3) == SVM_Y_Train(2000 +  1 : N_Train)) / (N_Train - 2000);
