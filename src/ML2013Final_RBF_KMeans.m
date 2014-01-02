% ---------- 清除舊資料 ----------
clear all;
close all;
clc;
% ---------- 計時開始 ----------
tic;
% ---------- 讀取資料 ----------
[SVM_Y_Train, SVM_X_Train] = libsvmread('ml2013final_train.dat');
[SVM_Y_Test, SVM_X_Test] = libsvmread('ml2013final_test1.nolabel.dat');
% ---------- 變數設定 ----------
[N_Train, Dimension_Train] = size(SVM_X_Train);
[N_Test, Dimension_Test] = size(SVM_X_Test);
[Total_Classes, ~] = max(SVM_Y_Train);
Center = 2;

N_Train = N_Train / (4 * 4);

% Acv
Indices = zeros(N_Train, 1);
Fold = 2;
N_Test_Acv = N_Train / Fold;
N_Test_Acv_Temp = 0;
Fold_Temp = 1;

for i = 1 : N_Train
    if (i <= N_Test_Acv * Fold_Temp)
        Indices(i, 1) = Fold_Temp;
        N_Test_Acv_Temp = N_Test_Acv_Temp + 1;
        if (N_Test_Acv == N_Test_Acv_Temp)
            N_Test_Acv_Temp = 0;
            Fold_Temp = Fold_Temp + 1;
        end
    end	
end

Class_01_X = [];
Class_02_X = [];
Class_03_X = [];
Class_04_X = [];
Class_05_X = [];
Class_06_X = [];
Class_07_X = [];
Class_08_X = [];
Class_09_X = [];
Class_10_X = [];
Class_11_X = [];
Class_12_X = [];

Class_01_Y = [];
Class_02_Y = [];
Class_03_Y = [];
Class_04_Y = [];
Class_05_Y = [];
Class_06_Y = [];
Class_07_Y = [];
Class_08_Y = [];
Class_09_Y = [];
Class_10_Y = [];
Class_11_Y = [];
Class_12_Y = [];

p = randperm(N_Train);

for i = 1 : Fold
	X_Test = SVM_X_Train(p(Indices == i), :);
	Y_Test = SVM_Y_Train(p(Indices == i), :);
	X_Train = SVM_X_Train(p(Indices ~= i), :);
	Y_Train = SVM_Y_Train(p(Indices ~= i), :);
	[N_Train, ~] = size(X_Train);
    [N_Test, ~] = size(X_Test);

    for j = 1 : N_Train
        switch Y_Train(j)
            case 1
                Class_01_X = [Class_01_X; X_Train(j, :)];
                Class_01_Y = [Class_01_Y; Y_Train(j, :)];
            case 2
                Class_02_X = [Class_02_X; X_Train(j, :)];
                Class_02_Y = [Class_02_Y; Y_Train(j, :)];
            case 3;
                Class_03_X = [Class_03_X; X_Train(j, :)];
                Class_03_Y = [Class_03_Y; Y_Train(j, :)];
            case 4
                Class_04_X = [Class_04_X; X_Train(j, :)];
                Class_04_Y = [Class_04_Y; Y_Train(j, :)];
            case 5
                Class_05_X = [Class_05_X; X_Train(j, :)];
                Class_05_Y = [Class_05_Y; Y_Train(j, :)];
            case 6
                Class_06_X = [Class_06_X; X_Train(j, :)];
                Class_06_Y = [Class_06_Y; Y_Train(j, :)];
            case 7
                Class_07_X = [Class_07_X; X_Train(j, :)];
                Class_07_Y = [Class_07_Y; Y_Train(j, :)];
            case 8
                Class_08_X = [Class_08_X; X_Train(j, :)];
                Class_08_Y = [Class_08_Y; Y_Train(j, :)];
            case 9
                Class_09_X = [Class_09_X; X_Train(j, :)];
                Class_09_Y = [Class_09_Y; Y_Train(j, :)];
            case 10
                Class_10_X = [Class_10_X; X_Train(j, :)];
                Class_10_Y = [Class_10_Y; Y_Train(j, :)];
            case 11
                Class_11_X = [Class_11_X; X_Train(j, :)];
                Class_11_Y = [Class_11_Y; Y_Train(j, :)];
            case 12
                Class_12_X = [Class_12_X; X_Train(j, :)];
                Class_12_Y = [Class_12_Y; Y_Train(j, :)];
        end
    end    

    Length_01 = size(Class_01_X, 1);
    Length_02 = size(Class_02_X, 1);
    Length_03 = size(Class_03_X, 1);
    Length_04 = size(Class_04_X, 1);
    Length_05 = size(Class_05_X, 1);
    Length_06 = size(Class_06_X, 1);
    Length_07 = size(Class_07_X, 1);
    Length_08 = size(Class_08_X, 1);
    Length_09 = size(Class_09_X, 1);
    Length_10 = size(Class_10_X, 1);
    Length_11 = size(Class_11_X, 1);
    Length_12 = size(Class_12_X, 1);
    
    % Combine 01
    Combine_01_02_X = [Class_01_X; Class_02_X];
    Combine_01_02_Y = [Class_01_Y; Class_02_Y];
    Combine_01_02_Y(1 : Length_01) = 1;
    Combine_01_02_Y((Length_01 + 1) : (Length_01 + Length_02)) = -1;
    
    Combine_01_03_X = [Class_01_X; Class_03_X];
    Combine_01_03_Y = [Class_01_Y; Class_03_Y];
    Combine_01_03_Y(1 : Length_01) = 1;
    Combine_01_03_Y((Length_01 + 1) : (Length_01 + Length_03)) = -1;
    
    Combine_01_04_X = [Class_01_X; Class_04_X];
    Combine_01_04_Y = [Class_01_Y; Class_04_Y];
    Combine_01_04_Y(1 : Length_01) = 1;
    Combine_01_04_Y((Length_01 + 1) : (Length_01 + Length_04)) = -1;
    
    Combine_01_05_X = [Class_01_X; Class_05_X];
    Combine_01_05_Y = [Class_01_Y; Class_05_Y];
    Combine_01_05_Y(1 : Length_01) = 1;
    Combine_01_05_Y((Length_01 + 1) : (Length_01 + Length_05)) = -1;
    
    Combine_01_06_X = [Class_01_X; Class_06_X];
    Combine_01_06_Y = [Class_01_Y; Class_06_Y];
    Combine_01_06_Y(1 : Length_01) = 1;
    Combine_01_06_Y((Length_01 + 1) : (Length_01 + Length_06)) = -1;
    
    Combine_01_07_X = [Class_01_X; Class_07_X];
    Combine_01_07_Y = [Class_01_Y; Class_07_Y];
    Combine_01_07_Y(1 : Length_01) = 1;
    Combine_01_07_Y((Length_01 + 1) : (Length_01 + Length_07)) = -1;
    
    Combine_01_08_X = [Class_01_X; Class_08_X];
    Combine_01_08_Y = [Class_01_Y; Class_08_Y];
    Combine_01_08_Y(1 : Length_01) = 1;
    Combine_01_08_Y((Length_01 + 1) : (Length_01 + Length_08)) = -1;
    
    Combine_01_09_X = [Class_01_X; Class_09_X];
    Combine_01_09_Y = [Class_01_Y; Class_09_Y];
    Combine_01_09_Y(1 : Length_01) = 1;
    Combine_01_09_Y((Length_01 + 1) : (Length_01 + Length_09)) = -1;
    
    Combine_01_10_X = [Class_01_X; Class_10_X];
    Combine_01_10_Y = [Class_01_Y; Class_10_Y];
    Combine_01_10_Y(1 : Length_01) = 1;
    Combine_01_10_Y((Length_01 + 1) : (Length_01 + Length_10)) = -1;
    
    Combine_01_11_X = [Class_01_X; Class_11_X];
    Combine_01_11_Y = [Class_01_Y; Class_11_Y];
    Combine_01_11_Y(1 : Length_01) = 1;
    Combine_01_11_Y((Length_01 + 1) : (Length_01 + Length_11)) = -1;
    
    Combine_01_12_X = [Class_01_X; Class_12_X];
    Combine_01_12_Y = [Class_01_Y; Class_12_Y];
    Combine_01_12_Y(1 : Length_01) = 1;
    Combine_01_12_Y((Length_01 + 1) : (Length_01 + Length_12)) = -1; 
    
    % Combine 02   
    Combine_02_01_X = [Class_02_X; Class_01_X];
    Combine_02_01_Y = [Class_02_Y; Class_01_Y];
    Combine_02_01_Y(1 : Length_02) = 1;
    Combine_02_01_Y((Length_02 + 1) : (Length_02 + Length_01)) = -1;
    
    Combine_02_03_X = [Class_02_X; Class_03_X];
    Combine_02_03_Y = [Class_02_Y; Class_03_Y];
    Combine_02_03_Y(1 : Length_02) = 1;
    Combine_02_03_Y((Length_02 + 1) : (Length_02 + Length_03)) = -1;
    
    Combine_02_04_X = [Class_02_X; Class_04_X];
    Combine_02_04_Y = [Class_02_Y; Class_04_Y];
    Combine_02_04_Y(1 : Length_02) = 1;
    Combine_02_04_Y((Length_02 + 1) : (Length_02 + Length_04)) = -1;
    
    Combine_02_05_X = [Class_02_X; Class_05_X];
    Combine_02_05_Y = [Class_02_Y; Class_05_Y];
    Combine_02_05_Y(1 : Length_02) = 1;
    Combine_02_05_Y((Length_02 + 1) : (Length_02 + Length_05)) = -1;
    
    Combine_02_06_X = [Class_02_X; Class_06_X];
    Combine_02_06_Y = [Class_02_Y; Class_06_Y];
    Combine_02_06_Y(1 : Length_02) = 1;
    Combine_02_06_Y((Length_02 + 1) : (Length_02 + Length_06)) = -1;
    
    Combine_02_07_X = [Class_02_X; Class_07_X];
    Combine_02_07_Y = [Class_02_Y; Class_07_Y];
    Combine_02_07_Y(1 : Length_02) = 1;
    Combine_02_07_Y((Length_02 + 1) : (Length_02 + Length_07)) = -1;
    
    Combine_02_08_X = [Class_02_X; Class_08_X];
    Combine_02_08_Y = [Class_02_Y; Class_08_Y];
    Combine_02_08_Y(1 : Length_02) = 1;
    Combine_02_08_Y((Length_02 + 1) : (Length_02 + Length_08)) = -1;
    
    Combine_02_09_X = [Class_02_X; Class_09_X];
    Combine_02_09_Y = [Class_02_Y; Class_09_Y];
    Combine_02_09_Y(1 : Length_02) = 1;
    Combine_02_09_Y((Length_02 + 1) : (Length_02 + Length_09)) = -1;
    
    Combine_02_10_X = [Class_02_X; Class_10_X];
    Combine_02_10_Y = [Class_02_Y; Class_10_Y];
    Combine_02_10_Y(1 : Length_02) = 1;
    Combine_02_10_Y((Length_02 + 1) : (Length_02 + Length_10)) = -1;
    
    Combine_02_11_X = [Class_02_X; Class_11_X];
    Combine_02_11_Y = [Class_02_Y; Class_11_Y];
    Combine_02_11_Y(1 : Length_02) = 1;
    Combine_02_11_Y((Length_02 + 1) : (Length_02 + Length_11)) = -1;
    
    Combine_02_12_X = [Class_02_X; Class_12_X];
    Combine_02_12_Y = [Class_02_Y; Class_12_Y];
    Combine_02_12_Y(1 : Length_02) = 1;
    Combine_02_12_Y((Length_02 + 1) : (Length_02 + Length_12)) = -1;
    
    % Combine 03
    Combine_03_01_X = [Class_03_X; Class_01_X];
    Combine_03_01_Y = [Class_03_Y; Class_01_Y];
    Combine_03_01_Y(1 : Length_03) = 1;
    Combine_03_01_Y((Length_03 + 1) : (Length_03 + Length_01)) = -1;
    
    Combine_03_02_X = [Class_03_X; Class_02_X];
    Combine_03_02_Y = [Class_03_Y; Class_02_Y];
    Combine_03_02_Y(1 : Length_03) = 1;
    Combine_03_02_Y((Length_03 + 1) : (Length_03 + Length_02)) = -1;
    
    Combine_03_04_X = [Class_03_X; Class_04_X];
    Combine_03_04_Y = [Class_03_Y; Class_04_Y];
    Combine_03_04_Y(1 : Length_03) = 1;
    Combine_03_04_Y((Length_03 + 1) : (Length_03 + Length_04)) = -1;
    
    Combine_03_05_X = [Class_03_X; Class_05_X];
    Combine_03_05_Y = [Class_03_Y; Class_05_Y];
    Combine_03_05_Y(1 : Length_03) = 1;
    Combine_03_05_Y((Length_03 + 1) : (Length_03 + Length_05)) = -1;
    
    Combine_03_06_X = [Class_03_X; Class_06_X];
    Combine_03_06_Y = [Class_03_Y; Class_06_Y];
    Combine_03_06_Y(1 : Length_03) = 1;
    Combine_03_06_Y((Length_03 + 1) : (Length_03 + Length_06)) = -1;
    
    Combine_03_07_X = [Class_03_X; Class_07_X];
    Combine_03_07_Y = [Class_03_Y; Class_07_Y];
    Combine_03_07_Y(1 : Length_03) = 1;
    Combine_03_07_Y((Length_03 + 1) : (Length_03 + Length_07)) = -1;
    
    Combine_03_08_X = [Class_03_X; Class_08_X];
    Combine_03_08_Y = [Class_03_Y; Class_08_Y];
    Combine_03_08_Y(1 : Length_03) = 1;
    Combine_03_08_Y((Length_03 + 1) : (Length_03 + Length_08)) = -1;
    
    Combine_03_09_X = [Class_03_X; Class_09_X];
    Combine_03_09_Y = [Class_03_Y; Class_09_Y];
    Combine_03_09_Y(1 : Length_03) = 1;
    Combine_03_09_Y((Length_03 + 1) : (Length_03 + Length_09)) = -1;
    
    Combine_03_10_X = [Class_03_X; Class_10_X];
    Combine_03_10_Y = [Class_03_Y; Class_10_Y];
    Combine_03_10_Y(1 : Length_03) = 1;
    Combine_03_10_Y((Length_03 + 1) : (Length_03 + Length_10)) = -1;
    
    Combine_03_11_X = [Class_03_X; Class_11_X];
    Combine_03_11_Y = [Class_03_Y; Class_11_Y];
    Combine_03_11_Y(1 : Length_03) = 1;
    Combine_03_11_Y((Length_03 + 1) : (Length_03 + Length_11)) = -1;
    
    Combine_03_12_X = [Class_03_X; Class_12_X];
    Combine_03_12_Y = [Class_03_Y; Class_12_Y];
    Combine_03_12_Y(1 : Length_03) = 1;
    Combine_03_12_Y((Length_03 + 1) : (Length_03 + Length_12)) = -1;
    
    % Combine 04
    Combine_04_01_X = [Class_04_X; Class_01_X];
    Combine_04_01_Y = [Class_04_Y; Class_01_Y];
    Combine_04_01_Y(1 : Length_04) = 1;
    Combine_04_01_Y((Length_04 + 1) : (Length_04 + Length_01)) = -1;
    
    Combine_04_02_X = [Class_04_X; Class_02_X];
    Combine_04_02_Y = [Class_04_Y; Class_02_Y];
    Combine_04_02_Y(1 : Length_04) = 1;
    Combine_04_02_Y((Length_04 + 1) : (Length_04 + Length_02)) = -1;
    
    Combine_04_03_X = [Class_04_X; Class_03_X];
    Combine_04_03_Y = [Class_04_Y; Class_03_Y];
    Combine_04_03_Y(1 : Length_04) = 1;
    Combine_04_03_Y((Length_04 + 1) : (Length_04 + Length_03)) = -1;
    
    Combine_04_05_X = [Class_04_X; Class_05_X];
    Combine_04_05_Y = [Class_04_Y; Class_05_Y];
    Combine_04_05_Y(1 : Length_04) = 1;
    Combine_04_05_Y((Length_04 + 1) : (Length_04 + Length_05)) = -1;
    
    Combine_04_06_X = [Class_04_X; Class_06_X];
    Combine_04_06_Y = [Class_04_Y; Class_06_Y];
    Combine_04_06_Y(1 : Length_04) = 1;
    Combine_04_06_Y((Length_04 + 1) : (Length_04 + Length_06)) = -1;
    
    Combine_04_07_X = [Class_04_X; Class_07_X];
    Combine_04_07_Y = [Class_04_Y; Class_07_Y];
    Combine_04_07_Y(1 : Length_04) = 1;
    Combine_04_07_Y((Length_04 + 1) : (Length_04 + Length_07)) = -1;
    
    Combine_04_08_X = [Class_04_X; Class_08_X];
    Combine_04_08_Y = [Class_04_Y; Class_08_Y];
    Combine_04_08_Y(1 : Length_04) = 1;
    Combine_04_08_Y((Length_04 + 1) : (Length_04 + Length_08)) = -1;
    
    Combine_04_09_X = [Class_04_X; Class_09_X];
    Combine_04_09_Y = [Class_04_Y; Class_09_Y];
    Combine_04_09_Y(1 : Length_04) = 1;
    Combine_04_09_Y((Length_04 + 1) : (Length_04 + Length_09)) = -1;
    
    Combine_04_10_X = [Class_04_X; Class_10_X];
    Combine_04_10_Y = [Class_04_Y; Class_10_Y];
    Combine_04_10_Y(1 : Length_04) = 1;
    Combine_04_10_Y((Length_04 + 1) : (Length_04 + Length_10)) = -1;
    
    Combine_04_11_X = [Class_04_X; Class_11_X];
    Combine_04_11_Y = [Class_04_Y; Class_11_Y];
    Combine_04_11_Y(1 : Length_04) = 1;
    Combine_04_11_Y((Length_04 + 1) : (Length_04 + Length_11)) = -1;
    
    Combine_04_12_X = [Class_04_X; Class_12_X];
    Combine_04_12_Y = [Class_04_Y; Class_12_Y];
    Combine_04_12_Y(1 : Length_04) = 1;
    Combine_04_12_Y((Length_04 + 1) : (Length_04 + Length_12)) = -1;
    
    % Combine 05
    Combine_05_01_X = [Class_05_X; Class_01_X];
    Combine_05_01_Y = [Class_05_Y; Class_01_Y];
    Combine_05_01_Y(1 : Length_05) = 1;
    Combine_05_01_Y((Length_05 + 1) : (Length_05 + Length_01)) = -1;
    
    Combine_05_02_X = [Class_05_X; Class_02_X];
    Combine_05_02_Y = [Class_05_Y; Class_02_Y];
    Combine_05_02_Y(1 : Length_05) = 1;
    Combine_05_02_Y((Length_05 + 1) : (Length_05 + Length_02)) = -1;
    
    Combine_05_03_X = [Class_05_X; Class_03_X];
    Combine_05_03_Y = [Class_05_Y; Class_03_Y];
    Combine_05_03_Y(1 : Length_05) = 1;
    Combine_05_03_Y((Length_05 + 1) : (Length_05 + Length_03)) = -1;
    
    Combine_05_04_X = [Class_05_X; Class_04_X];
    Combine_05_04_Y = [Class_05_Y; Class_04_Y];
    Combine_05_04_Y(1 : Length_05) = 1;
    Combine_05_04_Y((Length_05 + 1) : (Length_05 + Length_04)) = -1;
    
    Combine_05_06_X = [Class_05_X; Class_06_X];
    Combine_05_06_Y = [Class_05_Y; Class_06_Y];
    Combine_05_06_Y(1 : Length_05) = 1;
    Combine_05_06_Y((Length_05 + 1) : (Length_05 + Length_06)) = -1;
    
    Combine_05_07_X = [Class_05_X; Class_07_X];
    Combine_05_07_Y = [Class_05_Y; Class_07_Y];
    Combine_05_07_Y(1 : Length_05) = 1;
    Combine_05_07_Y((Length_05 + 1) : (Length_05 + Length_07)) = -1;
    
    Combine_05_08_X = [Class_05_X; Class_08_X];
    Combine_05_08_Y = [Class_05_Y; Class_08_Y];
    Combine_05_08_Y(1 : Length_05) = 1;
    Combine_05_08_Y((Length_05 + 1) : (Length_05 + Length_08)) = -1;
    
    Combine_05_09_X = [Class_05_X; Class_09_X];
    Combine_05_09_Y = [Class_05_Y; Class_09_Y];
    Combine_05_09_Y(1 : Length_05) = 1;
    Combine_05_09_Y((Length_05 + 1) : (Length_05 + Length_09)) = -1;
    
    Combine_05_10_X = [Class_05_X; Class_10_X];
    Combine_05_10_Y = [Class_05_Y; Class_10_Y];
    Combine_05_10_Y(1 : Length_05) = 1;
    Combine_05_10_Y((Length_05 + 1) : (Length_05 + Length_10)) = -1;
    
    Combine_05_11_X = [Class_05_X; Class_11_X];
    Combine_05_11_Y = [Class_05_Y; Class_11_Y];
    Combine_05_11_Y(1 : Length_05) = 1;
    Combine_05_11_Y((Length_05 + 1) : (Length_05 + Length_11)) = -1;
    
    Combine_05_12_X = [Class_05_X; Class_12_X];
    Combine_05_12_Y = [Class_05_Y; Class_12_Y];
    Combine_05_12_Y(1 : Length_05) = 1;
    Combine_05_12_Y((Length_05 + 1) : (Length_05 + Length_12)) = -1;
    
    % Combine 06
    Combine_06_01_X = [Class_06_X; Class_01_X];
    Combine_06_01_Y = [Class_06_Y; Class_01_Y];
    Combine_06_01_Y(1 : Length_06) = 1;
    Combine_06_01_Y((Length_06 + 1) : (Length_06 + Length_01)) = -1;
    
    Combine_06_02_X = [Class_06_X; Class_02_X];
    Combine_06_02_Y = [Class_06_Y; Class_02_Y];
    Combine_06_02_Y(1 : Length_06) = 1;
    Combine_06_02_Y((Length_06 + 1) : (Length_06 + Length_02)) = -1;
    
    Combine_06_03_X = [Class_06_X; Class_03_X];
    Combine_06_03_Y = [Class_06_Y; Class_03_Y];
    Combine_06_03_Y(1 : Length_06) = 1;
    Combine_06_03_Y((Length_06 + 1) : (Length_06 + Length_03)) = -1;
    
    Combine_06_04_X = [Class_06_X; Class_04_X];
    Combine_06_04_Y = [Class_06_Y; Class_04_Y];
    Combine_06_04_Y(1 : Length_06) = 1;
    Combine_06_04_Y((Length_06 + 1) : (Length_06 + Length_04)) = -1;
    
    Combine_06_05_X = [Class_06_X; Class_05_X];
    Combine_06_05_Y = [Class_06_Y; Class_05_Y];
    Combine_06_05_Y(1 : Length_06) = 1;
    Combine_06_05_Y((Length_06 + 1) : (Length_06 + Length_05)) = -1;
    
    Combine_06_07_X = [Class_06_X; Class_07_X];
    Combine_06_07_Y = [Class_06_Y; Class_07_Y];
    Combine_06_07_Y(1 : Length_06) = 1;
    Combine_06_07_Y((Length_06 + 1) : (Length_06 + Length_07)) = -1;
    
    Combine_06_08_X = [Class_06_X; Class_08_X];
    Combine_06_08_Y = [Class_06_Y; Class_08_Y];
    Combine_06_08_Y(1 : Length_06) = 1;
    Combine_06_08_Y((Length_06 + 1) : (Length_06 + Length_08)) = -1;
    
    Combine_06_09_X = [Class_06_X; Class_09_X];
    Combine_06_09_Y = [Class_06_Y; Class_09_Y];
    Combine_06_09_Y(1 : Length_06) = 1;
    Combine_06_09_Y((Length_06 + 1) : (Length_06 + Length_09)) = -1;
    
    Combine_06_10_X = [Class_06_X; Class_10_X];
    Combine_06_10_Y = [Class_06_Y; Class_10_Y];
    Combine_06_10_Y(1 : Length_06) = 1;
    Combine_06_10_Y((Length_06 + 1) : (Length_06 + Length_10)) = -1;
    
    Combine_06_11_X = [Class_06_X; Class_11_X];
    Combine_06_11_Y = [Class_06_Y; Class_11_Y];
    Combine_06_11_Y(1 : Length_06) = 1;
    Combine_06_11_Y((Length_06 + 1) : (Length_06 + Length_11)) = -1;
    
    Combine_06_12_X = [Class_06_X; Class_12_X];
    Combine_06_12_Y = [Class_06_Y; Class_12_Y];
    Combine_06_12_Y(1 : Length_06) = 1;
    Combine_06_12_Y((Length_06 + 1) : (Length_06 + Length_12)) = -1;
    
    % Combine 07
    Combine_07_01_X = [Class_07_X; Class_01_X];
    Combine_07_01_Y = [Class_07_Y; Class_01_Y];
    Combine_07_01_Y(1 : Length_07) = 1;
    Combine_07_01_Y((Length_07 + 1) : (Length_07 + Length_01)) = -1;
    
    Combine_07_02_X = [Class_07_X; Class_02_X];
    Combine_07_02_Y = [Class_07_Y; Class_02_Y];
    Combine_07_02_Y(1 : Length_07) = 1;
    Combine_07_02_Y((Length_07 + 1) : (Length_07 + Length_02)) = -1;
    
    Combine_07_03_X = [Class_07_X; Class_03_X];
    Combine_07_03_Y = [Class_07_Y; Class_03_Y];
    Combine_07_03_Y(1 : Length_07) = 1;
    Combine_07_03_Y((Length_07 + 1) : (Length_07 + Length_03)) = -1;
    
    Combine_07_04_X = [Class_07_X; Class_04_X];
    Combine_07_04_Y = [Class_07_Y; Class_04_Y];
    Combine_07_04_Y(1 : Length_07) = 1;
    Combine_07_04_Y((Length_07 + 1) : (Length_07 + Length_04)) = -1;
    
    Combine_07_05_X = [Class_07_X; Class_05_X];
    Combine_07_05_Y = [Class_07_Y; Class_05_Y];
    Combine_07_05_Y(1 : Length_07) = 1;
    Combine_07_05_Y((Length_07 + 1) : (Length_07 + Length_05)) = -1;
    
	Combine_07_06_X = [Class_07_X; Class_06_X];
    Combine_07_06_Y = [Class_07_Y; Class_06_Y];
    Combine_07_06_Y(1 : Length_07) = 1;
    Combine_07_06_Y((Length_07 + 1) : (Length_07 + Length_06)) = -1;
    
	Combine_07_08_X = [Class_07_X; Class_08_X];
    Combine_07_08_Y = [Class_07_Y; Class_08_Y];
    Combine_07_08_Y(1 : Length_07) = 1;
    Combine_07_08_Y((Length_07 + 1) : (Length_07 + Length_08)) = -1;
    
    Combine_07_09_X = [Class_07_X; Class_09_X];
    Combine_07_09_Y = [Class_07_Y; Class_09_Y];
    Combine_07_09_Y(1 : Length_07) = 1;
    Combine_07_09_Y((Length_07 + 1) : (Length_07 + Length_09)) = -1;
    
    Combine_07_10_X = [Class_07_X; Class_10_X];
    Combine_07_10_Y = [Class_07_Y; Class_10_Y];
    Combine_07_10_Y(1 : Length_07) = 1;
    Combine_07_10_Y((Length_07 + 1) : (Length_07 + Length_10)) = -1;
    
    Combine_07_11_X = [Class_07_X; Class_11_X];
    Combine_07_11_Y = [Class_07_Y; Class_11_Y];
    Combine_07_11_Y(1 : Length_07) = 1;
    Combine_07_11_Y((Length_07 + 1) : (Length_07 + Length_11)) = -1;
    
    Combine_07_12_X = [Class_07_X; Class_12_X];
    Combine_07_12_Y = [Class_07_Y; Class_12_Y];
    Combine_07_12_Y(1 : Length_07) = 1;
    Combine_07_12_Y((Length_07 + 1) : (Length_07 + Length_12)) = -1;
    
    % Combine 08
    Combine_08_01_X = [Class_08_X; Class_01_X];
    Combine_08_01_Y = [Class_08_Y; Class_01_Y];
    Combine_08_01_Y(1 : Length_08) = 1;
    Combine_08_01_Y((Length_08 + 1) : (Length_08 + Length_01)) = -1;
    
    Combine_08_02_X = [Class_08_X; Class_02_X];
    Combine_08_02_Y = [Class_08_Y; Class_02_Y];
    Combine_08_02_Y(1 : Length_08) = 1;
    Combine_08_02_Y((Length_08 + 1) : (Length_08 + Length_02)) = -1;
    
    Combine_08_03_X = [Class_08_X; Class_03_X];
    Combine_08_03_Y = [Class_08_Y; Class_03_Y];
    Combine_08_03_Y(1 : Length_08) = 1;
    Combine_08_03_Y((Length_08 + 1) : (Length_08 + Length_03)) = -1;
    
    Combine_08_04_X = [Class_08_X; Class_04_X];
    Combine_08_04_Y = [Class_08_Y; Class_04_Y];
    Combine_08_04_Y(1 : Length_08) = 1;
    Combine_08_04_Y((Length_08 + 1) : (Length_08 + Length_04)) = -1;
    
    Combine_08_05_X = [Class_08_X; Class_05_X];
    Combine_08_05_Y = [Class_08_Y; Class_05_Y];
    Combine_08_05_Y(1 : Length_08) = 1;
    Combine_08_05_Y((Length_08 + 1) : (Length_08 + Length_05)) = -1;
    
    Combine_08_06_X = [Class_08_X; Class_06_X];
    Combine_08_06_Y = [Class_08_Y; Class_06_Y];
    Combine_08_06_Y(1 : Length_08) = 1;
    Combine_08_06_Y((Length_08 + 1) : (Length_08 + Length_06)) = -1;
    
    Combine_08_07_X = [Class_08_X; Class_07_X];
    Combine_08_07_Y = [Class_08_Y; Class_07_Y];
    Combine_08_07_Y(1 : Length_08) = 1;
    Combine_08_07_Y((Length_08 + 1) : (Length_08 + Length_07)) = -1;
    
    Combine_08_09_X = [Class_08_X; Class_09_X];
    Combine_08_09_Y = [Class_08_Y; Class_09_Y];
    Combine_08_09_Y(1 : Length_08) = 1;
    Combine_08_09_Y((Length_08 + 1) : (Length_08 + Length_09)) = -1;
    
    Combine_08_10_X = [Class_08_X; Class_10_X];
    Combine_08_10_Y = [Class_08_Y; Class_10_Y];
    Combine_08_10_Y(1 : Length_08) = 1;
    Combine_08_10_Y((Length_08 + 1) : (Length_08 + Length_10)) = -1;
    
    Combine_08_11_X = [Class_08_X; Class_11_X];
    Combine_08_11_Y = [Class_08_Y; Class_11_Y];
    Combine_08_11_Y(1 : Length_08) = 1;
    Combine_08_11_Y((Length_08 + 1) : (Length_08 + Length_11)) = -1;
    
    Combine_08_12_X = [Class_08_X; Class_12_X];
    Combine_08_12_Y = [Class_08_Y; Class_12_Y];
    Combine_08_12_Y(1 : Length_08) = 1;
    Combine_08_12_Y((Length_08 + 1) : (Length_08 + Length_12)) = -1;
    
    % Combine 09   
    Combine_09_01_X = [Class_09_X; Class_01_X];
    Combine_09_01_Y = [Class_09_Y; Class_01_Y];
    Combine_09_01_Y(1 : Length_09) = 1;
    Combine_09_01_Y((Length_09 + 1) : (Length_09 + Length_01)) = -1;
    
    Combine_09_02_X = [Class_09_X; Class_02_X];
    Combine_09_02_Y = [Class_09_Y; Class_02_Y];
    Combine_09_02_Y(1 : Length_09) = 1;
    Combine_09_02_Y((Length_09 + 1) : (Length_09 + Length_02)) = -1;
    
    Combine_09_03_X = [Class_09_X; Class_03_X];
    Combine_09_03_Y = [Class_09_Y; Class_03_Y];
    Combine_09_03_Y(1 : Length_09) = 1;
    Combine_09_03_Y((Length_09 + 1) : (Length_09 + Length_03)) = -1;
    
    Combine_09_04_X = [Class_09_X; Class_04_X];
    Combine_09_04_Y = [Class_09_Y; Class_04_Y];
    Combine_09_04_Y(1 : Length_09) = 1;
    Combine_09_04_Y((Length_09 + 1) : (Length_09 + Length_04)) = -1;
    
    Combine_09_05_X = [Class_09_X; Class_05_X];
    Combine_09_05_Y = [Class_09_Y; Class_05_Y];
    Combine_09_05_Y(1 : Length_09) = 1;
    Combine_09_05_Y((Length_09 + 1) : (Length_09 + Length_05)) = -1;
    
    Combine_09_06_X = [Class_09_X; Class_06_X];
    Combine_09_06_Y = [Class_09_Y; Class_06_Y];
    Combine_09_06_Y(1 : Length_09) = 1;
    Combine_09_06_Y((Length_09 + 1) : (Length_09 + Length_06)) = -1;
    
    Combine_09_07_X = [Class_09_X; Class_07_X];
    Combine_09_07_Y = [Class_09_Y; Class_07_Y];
    Combine_09_07_Y(1 : Length_09) = 1;
    Combine_09_07_Y((Length_09 + 1) : (Length_09 + Length_07)) = -1;
    
    Combine_09_08_X = [Class_09_X; Class_08_X];
    Combine_09_08_Y = [Class_09_Y; Class_08_Y];
    Combine_09_08_Y(1 : Length_09) = 1;
    Combine_09_08_Y((Length_09 + 1) : (Length_09 + Length_08)) = -1;
    
    Combine_09_10_X = [Class_09_X; Class_10_X];
    Combine_09_10_Y = [Class_09_Y; Class_10_Y];
    Combine_09_10_Y(1 : Length_09) = 1;
    Combine_09_10_Y((Length_09 + 1) : (Length_09 + Length_10)) = -1;
    
    Combine_09_11_X = [Class_09_X; Class_11_X];
    Combine_09_11_Y = [Class_09_Y; Class_11_Y];
    Combine_09_11_Y(1 : Length_09) = 1;
    Combine_09_11_Y((Length_09 + 1) : (Length_09 + Length_11)) = -1;
    
    Combine_09_12_X = [Class_09_X; Class_12_X];
    Combine_09_12_Y = [Class_09_Y; Class_12_Y];
    Combine_09_12_Y(1 : Length_09) = 1;
    Combine_09_12_Y((Length_09 + 1) : (Length_09 + Length_12)) = -1;
    
    % Combine 10
    Combine_10_01_X = [Class_10_X; Class_01_X];
    Combine_10_01_Y = [Class_10_Y; Class_01_Y];
    Combine_10_01_Y(1 : Length_10) = 1;
    Combine_10_01_Y((Length_10 + 1) : (Length_10 + Length_01)) = -1;
    
    Combine_10_02_X = [Class_10_X; Class_02_X];
    Combine_10_02_Y = [Class_10_Y; Class_02_Y];
    Combine_10_02_Y(1 : Length_10) = 1;
    Combine_10_02_Y((Length_10 + 1) : (Length_10 + Length_02)) = -1;
    
    Combine_10_03_X = [Class_10_X; Class_03_X];
    Combine_10_03_Y = [Class_10_Y; Class_03_Y];
    Combine_10_03_Y(1 : Length_10) = 1;
    Combine_10_03_Y((Length_10 + 1) : (Length_10 + Length_03)) = -1;
    
    Combine_10_04_X = [Class_10_X; Class_04_X];
    Combine_10_04_Y = [Class_10_Y; Class_04_Y];
    Combine_10_04_Y(1 : Length_10) = 1;
    Combine_10_04_Y((Length_10 + 1) : (Length_10 + Length_04)) = -1;
    
    Combine_10_05_X = [Class_10_X; Class_05_X];
    Combine_10_05_Y = [Class_10_Y; Class_05_Y];
    Combine_10_05_Y(1 : Length_10) = 1;
    Combine_10_05_Y((Length_10 + 1) : (Length_10 + Length_05)) = -1;
    
    Combine_10_06_X = [Class_10_X; Class_06_X];
    Combine_10_06_Y = [Class_10_Y; Class_06_Y];
    Combine_10_06_Y(1 : Length_10) = 1;
    Combine_10_06_Y((Length_10 + 1) : (Length_10 + Length_06)) = -1;
    
    Combine_10_07_X = [Class_10_X; Class_07_X];
    Combine_10_07_Y = [Class_10_Y; Class_07_Y];
    Combine_10_07_Y(1 : Length_10) = 1;
    Combine_10_07_Y((Length_10 + 1) : (Length_10 + Length_07)) = -1;
    
    Combine_10_08_X = [Class_10_X; Class_08_X];
    Combine_10_08_Y = [Class_10_Y; Class_08_Y];
    Combine_10_08_Y(1 : Length_10) = 1;
    Combine_10_08_Y((Length_10 + 1) : (Length_10 + Length_08)) = -1;
    
    Combine_10_09_X = [Class_10_X; Class_09_X];
    Combine_10_09_Y = [Class_10_Y; Class_09_Y];
    Combine_10_09_Y(1 : Length_10) = 1;
    Combine_10_09_Y((Length_10 + 1) : (Length_10 + Length_09)) = -1;
    
    Combine_10_11_X = [Class_10_X; Class_11_X];
    Combine_10_11_Y = [Class_10_Y; Class_11_Y];
    Combine_10_11_Y(1 : Length_10) = 1;
    Combine_10_11_Y((Length_10 + 1) : (Length_10 + Length_11)) = -1;
    
    Combine_10_12_X = [Class_10_X; Class_12_X];
    Combine_10_12_Y = [Class_10_Y; Class_12_Y];
    Combine_10_12_Y(1 : Length_10) = 1;
    Combine_10_12_Y((Length_10 + 1) : (Length_10 + Length_12)) = -1;
    
    % Combine 11
    Combine_11_01_X = [Class_11_X; Class_01_X];
    Combine_11_01_Y = [Class_11_Y; Class_01_Y];
    Combine_11_01_Y(1 : Length_11) = 1;
    Combine_11_01_Y((Length_11 + 1) : (Length_11 + Length_01)) = -1;
    
    Combine_11_02_X = [Class_11_X; Class_02_X];
    Combine_11_02_Y = [Class_11_Y; Class_02_Y];
    Combine_11_02_Y(1 : Length_11) = 1;
    Combine_11_02_Y((Length_11 + 1) : (Length_11 + Length_02)) = -1;
    
    Combine_11_03_X = [Class_11_X; Class_03_X];
    Combine_11_03_Y = [Class_11_Y; Class_03_Y];
    Combine_11_03_Y(1 : Length_11) = 1;
    Combine_11_03_Y((Length_11 + 1) : (Length_11 + Length_03)) = -1;
    
    Combine_11_04_X = [Class_11_X; Class_04_X];
    Combine_11_04_Y = [Class_11_Y; Class_04_Y];
    Combine_11_04_Y(1 : Length_11) = 1;
    Combine_11_04_Y((Length_11 + 1) : (Length_11 + Length_04)) = -1;
    
    Combine_11_05_X = [Class_11_X; Class_05_X];
    Combine_11_05_Y = [Class_11_Y; Class_05_Y];
    Combine_11_05_Y(1 : Length_11) = 1;
    Combine_11_05_Y((Length_11 + 1) : (Length_11 + Length_05)) = -1;
    
    Combine_11_06_X = [Class_11_X; Class_06_X];
    Combine_11_06_Y = [Class_11_Y; Class_06_Y];
    Combine_11_06_Y(1 : Length_11) = 1;
    Combine_11_06_Y((Length_11 + 1) : (Length_11 + Length_06)) = -1;
    
    Combine_11_07_X = [Class_11_X; Class_07_X];
    Combine_11_07_Y = [Class_11_Y; Class_07_Y];
    Combine_11_07_Y(1 : Length_11) = 1;
    Combine_11_07_Y((Length_11 + 1) : (Length_11 + Length_07)) = -1;
    
    Combine_11_08_X = [Class_11_X; Class_08_X];
    Combine_11_08_Y = [Class_11_Y; Class_08_Y];
    Combine_11_08_Y(1 : Length_11) = 1;
    Combine_11_08_Y((Length_11 + 1) : (Length_11 + Length_08)) = -1;
    
    Combine_11_09_X = [Class_11_X; Class_09_X];
    Combine_11_09_Y = [Class_11_Y; Class_09_Y];
    Combine_11_09_Y(1 : Length_11) = 1;
    Combine_11_09_Y((Length_11 + 1) : (Length_11 + Length_09)) = -1;
    
    Combine_11_10_X = [Class_11_X; Class_10_X];
    Combine_11_10_Y = [Class_11_Y; Class_10_Y];
    Combine_11_10_Y(1 : Length_11) = 1;
    Combine_11_10_Y((Length_11 + 1) : (Length_11 + Length_10)) = -1;
    
    Combine_11_12_X = [Class_11_X; Class_12_X];
    Combine_11_12_Y = [Class_11_Y; Class_12_Y];
    Combine_11_12_Y(1 : Length_11) = 1;
    Combine_11_12_Y((Length_11 + 1) : (Length_11 + Length_12)) = -1;
    
    % Combine 12
    Combine_12_01_X = [Class_12_X; Class_01_X];
    Combine_12_01_Y = [Class_12_Y; Class_01_Y];
    Combine_12_01_Y(1 : Length_12) = 1;
    Combine_12_01_Y((Length_12 + 1) : (Length_12 + Length_01)) = -1;
    
    Combine_12_02_X = [Class_12_X; Class_02_X];
    Combine_12_02_Y = [Class_12_Y; Class_02_Y];
    Combine_12_02_Y(1 : Length_12) = 1;
    Combine_12_02_Y((Length_12 + 1) : (Length_12 + Length_02)) = -1;
    
    Combine_12_03_X = [Class_12_X; Class_03_X];
    Combine_12_03_Y = [Class_12_Y; Class_03_Y];
    Combine_12_03_Y(1 : Length_12) = 1;
    Combine_12_03_Y((Length_12 + 1) : (Length_12 + Length_03)) = -1;
    
    Combine_12_04_X = [Class_12_X; Class_04_X];
    Combine_12_04_Y = [Class_12_Y; Class_04_Y];
    Combine_12_04_Y(1 : Length_12) = 1;
    Combine_12_04_Y((Length_12 + 1) : (Length_12 + Length_04)) = -1;
    
    Combine_12_05_X = [Class_12_X; Class_05_X];
    Combine_12_05_Y = [Class_12_Y; Class_05_Y];
    Combine_12_05_Y(1 : Length_12) = 1;
    Combine_12_05_Y((Length_12 + 1) : (Length_12 + Length_05)) = -1;
    
    Combine_12_06_X = [Class_12_X; Class_06_X];
    Combine_12_06_Y = [Class_12_Y; Class_06_Y];
    Combine_12_06_Y(1 : Length_12) = 1;
    Combine_12_06_Y((Length_12 + 1) : (Length_12 + Length_06)) = -1;
    
    Combine_12_07_X = [Class_12_X; Class_07_X];
    Combine_12_07_Y = [Class_12_Y; Class_07_Y];
    Combine_12_07_Y(1 : Length_12) = 1;
    Combine_12_07_Y((Length_12 + 1) : (Length_12 + Length_07)) = -1;
    
    Combine_12_08_X = [Class_12_X; Class_08_X];
    Combine_12_08_Y = [Class_12_Y; Class_08_Y];
    Combine_12_08_Y(1 : Length_12) = 1;
    Combine_12_08_Y((Length_12 + 1) : (Length_12 + Length_08)) = -1;
    
    Combine_12_09_X = [Class_12_X; Class_09_X];
    Combine_12_09_Y = [Class_12_Y; Class_09_Y];
    Combine_12_09_Y(1 : Length_12) = 1;
    Combine_12_09_Y((Length_12 + 1) : (Length_12 + Length_09)) = -1;
    
    Combine_12_10_X = [Class_12_X; Class_10_X];
    Combine_12_10_Y = [Class_12_Y; Class_10_Y];
    Combine_12_10_Y(1 : Length_12) = 1;
    Combine_12_10_Y((Length_12 + 1) : (Length_12 + Length_10)) = -1;
    
    Combine_12_11_X = [Class_12_X; Class_11_X];
    Combine_12_11_Y = [Class_12_Y; Class_11_Y];
    Combine_12_11_Y(1 : Length_12) = 1;
    Combine_12_11_Y((Length_12 + 1) : (Length_12 + Length_11)) = -1;
    
    
    % Model 01
    disp('---------- Model 01 Start ----------');
    [newcenter_01_02, sigma_01_02, W_01_02, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_01_02_X, Combine_01_02_Y, Center);
    [newcenter_01_03, sigma_01_03, W_01_03, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_01_03_X, Combine_01_03_Y, Center);
    [newcenter_01_04, sigma_01_04, W_01_04, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_01_04_X, Combine_01_04_Y, Center);
    [newcenter_01_05, sigma_01_05, W_01_05, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_01_05_X, Combine_01_05_Y, Center);
    [newcenter_01_06, sigma_01_06, W_01_06, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_01_06_X, Combine_01_06_Y, Center);
    [newcenter_01_07, sigma_01_07, W_01_07, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_01_07_X, Combine_01_07_Y, Center);
    [newcenter_01_08, sigma_01_08, W_01_08, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_01_08_X, Combine_01_08_Y, Center);
    [newcenter_01_09, sigma_01_09, W_01_09, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_01_09_X, Combine_01_09_Y, Center);
    [newcenter_01_10, sigma_01_10, W_01_10, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_01_10_X, Combine_01_10_Y, Center);
    [newcenter_01_11, sigma_01_11, W_01_11, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_01_11_X, Combine_01_11_Y, Center);
    [newcenter_01_12, sigma_01_12, W_01_12, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_01_12_X, Combine_01_12_Y, Center);    
    disp('---------- Model 01 Over ----------');
    
    % Model 02
    disp('---------- Model 02 Start ----------');
    [newcenter_02_01, sigma_02_01, W_02_01, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_02_01_X, Combine_02_01_Y, Center);
    [newcenter_02_03, sigma_02_03, W_02_03, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_02_03_X, Combine_02_03_Y, Center);
    [newcenter_02_04, sigma_02_04, W_02_04, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_02_04_X, Combine_02_04_Y, Center);
    [newcenter_02_05, sigma_02_05, W_02_05, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_02_05_X, Combine_02_05_Y, Center);
    [newcenter_02_06, sigma_02_06, W_02_06, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_02_06_X, Combine_02_06_Y, Center);
    [newcenter_02_07, sigma_02_07, W_02_07, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_02_07_X, Combine_02_07_Y, Center);
    [newcenter_02_08, sigma_02_08, W_02_08, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_02_08_X, Combine_02_08_Y, Center);
    [newcenter_02_09, sigma_02_09, W_02_09, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_02_09_X, Combine_02_09_Y, Center);
    [newcenter_02_10, sigma_02_10, W_02_10, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_02_10_X, Combine_02_10_Y, Center);
    [newcenter_02_11, sigma_02_11, W_02_11, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_02_11_X, Combine_02_11_Y, Center);
    [newcenter_02_12, sigma_02_12, W_02_12, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_02_12_X, Combine_02_12_Y, Center);    
    disp('---------- Model 02 Over ----------');
    
    % Model 03
    disp('---------- Model 03 Start ----------');
    [newcenter_03_01, sigma_03_01, W_03_01, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_03_01_X, Combine_03_01_Y, Center);
    [newcenter_03_02, sigma_03_02, W_03_02, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_03_02_X, Combine_03_02_Y, Center);
    [newcenter_03_04, sigma_03_04, W_03_04, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_03_04_X, Combine_03_04_Y, Center);
    [newcenter_03_05, sigma_03_05, W_03_05, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_03_05_X, Combine_03_05_Y, Center);
    [newcenter_03_06, sigma_03_06, W_03_06, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_03_06_X, Combine_03_06_Y, Center);
    [newcenter_03_07, sigma_03_07, W_03_07, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_03_07_X, Combine_03_07_Y, Center);
    [newcenter_03_08, sigma_03_08, W_03_08, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_03_08_X, Combine_03_08_Y, Center);
    [newcenter_03_09, sigma_03_09, W_03_09, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_03_09_X, Combine_03_09_Y, Center);
    [newcenter_03_10, sigma_03_10, W_03_10, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_03_10_X, Combine_03_10_Y, Center);
    [newcenter_03_11, sigma_03_11, W_03_11, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_03_11_X, Combine_03_11_Y, Center);
    [newcenter_03_12, sigma_03_12, W_03_12, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_03_12_X, Combine_03_12_Y, Center);      
    disp('---------- Model 03 Over ----------');
    
    % Model 04
    disp('---------- Model 04 Start ----------');
    [newcenter_04_01, sigma_04_01, W_04_01, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_04_01_X, Combine_04_01_Y, Center);
    [newcenter_04_02, sigma_04_02, W_04_02, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_04_02_X, Combine_04_02_Y, Center);
    [newcenter_04_03, sigma_04_03, W_04_03, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_04_03_X, Combine_04_03_Y, Center);
    [newcenter_04_05, sigma_04_05, W_04_05, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_04_05_X, Combine_04_05_Y, Center);
    [newcenter_04_06, sigma_04_06, W_04_06, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_04_06_X, Combine_04_06_Y, Center);
    [newcenter_04_07, sigma_04_07, W_04_07, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_04_07_X, Combine_04_07_Y, Center);
    [newcenter_04_08, sigma_04_08, W_04_08, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_04_08_X, Combine_04_08_Y, Center);
    [newcenter_04_09, sigma_04_09, W_04_09, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_04_09_X, Combine_04_09_Y, Center);
    [newcenter_04_10, sigma_04_10, W_04_10, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_04_10_X, Combine_04_10_Y, Center);
    [newcenter_04_11, sigma_04_11, W_04_11, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_04_11_X, Combine_04_11_Y, Center);
    [newcenter_04_12, sigma_04_12, W_04_12, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_04_12_X, Combine_04_12_Y, Center);   
    disp('---------- Model 04 Over ----------');
    
    % Model 05
    disp('---------- Model 05 Start ----------');
    [newcenter_05_01, sigma_05_01, W_05_01, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_05_01_X, Combine_05_01_Y, Center);
    [newcenter_05_02, sigma_05_02, W_05_02, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_05_02_X, Combine_05_02_Y, Center);
    [newcenter_05_03, sigma_05_03, W_05_03, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_05_03_X, Combine_05_03_Y, Center);
    [newcenter_05_04, sigma_05_04, W_05_04, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_05_04_X, Combine_05_04_Y, Center);
    [newcenter_05_06, sigma_05_06, W_05_06, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_05_06_X, Combine_05_06_Y, Center);
    [newcenter_05_07, sigma_05_07, W_05_07, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_05_07_X, Combine_05_07_Y, Center);
    [newcenter_05_08, sigma_05_08, W_05_08, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_05_08_X, Combine_05_08_Y, Center);
    [newcenter_05_09, sigma_05_09, W_05_09, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_05_09_X, Combine_05_09_Y, Center);
    [newcenter_05_10, sigma_05_10, W_05_10, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_05_10_X, Combine_05_10_Y, Center);
    [newcenter_05_11, sigma_05_11, W_05_11, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_05_11_X, Combine_05_11_Y, Center);
    [newcenter_05_12, sigma_05_12, W_05_12, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_05_12_X, Combine_05_12_Y, Center);   
    disp('---------- Model 05 Over ----------');
    
    % Model 06
    disp('---------- Model 06 Start ----------');
    [newcenter_06_01, sigma_06_01, W_06_01, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_06_01_X, Combine_06_01_Y, Center);
    [newcenter_06_02, sigma_06_02, W_06_02, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_06_02_X, Combine_06_02_Y, Center);
    [newcenter_06_03, sigma_06_03, W_06_03, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_06_03_X, Combine_06_03_Y, Center);
    [newcenter_06_04, sigma_06_04, W_06_04, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_06_04_X, Combine_06_04_Y, Center);
    [newcenter_06_05, sigma_06_05, W_06_05, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_06_05_X, Combine_06_05_Y, Center);
    [newcenter_06_07, sigma_06_07, W_06_07, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_06_07_X, Combine_06_07_Y, Center);
    [newcenter_06_08, sigma_06_08, W_06_08, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_06_08_X, Combine_06_08_Y, Center);
    [newcenter_06_09, sigma_06_09, W_06_09, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_06_09_X, Combine_06_09_Y, Center);
    [newcenter_06_10, sigma_06_10, W_06_10, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_06_10_X, Combine_06_10_Y, Center);
    [newcenter_06_11, sigma_06_11, W_06_11, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_06_11_X, Combine_06_11_Y, Center);
    [newcenter_06_12, sigma_06_12, W_06_12, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_06_12_X, Combine_06_12_Y, Center);   
    disp('---------- Model 06 Over ----------');
    
    % Model 07
    disp('---------- Model 07 Start ----------');
    [newcenter_07_01, sigma_07_01, W_07_01, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_07_01_X, Combine_07_01_Y, Center);
    [newcenter_07_02, sigma_07_02, W_07_02, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_07_02_X, Combine_07_02_Y, Center);
    [newcenter_07_03, sigma_07_03, W_07_03, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_07_03_X, Combine_07_03_Y, Center);
    [newcenter_07_04, sigma_07_04, W_07_04, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_07_04_X, Combine_07_04_Y, Center);
    [newcenter_07_05, sigma_07_05, W_07_05, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_07_05_X, Combine_07_05_Y, Center);
    [newcenter_07_06, sigma_07_06, W_07_06, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_07_06_X, Combine_07_06_Y, Center);
    [newcenter_07_08, sigma_07_08, W_07_08, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_07_08_X, Combine_07_08_Y, Center);
    [newcenter_07_09, sigma_07_09, W_07_09, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_07_09_X, Combine_07_09_Y, Center);
    [newcenter_07_10, sigma_07_10, W_07_10, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_07_10_X, Combine_07_10_Y, Center);
    [newcenter_07_11, sigma_07_11, W_07_11, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_07_11_X, Combine_07_11_Y, Center);
    [newcenter_07_12, sigma_07_12, W_07_12, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_07_12_X, Combine_07_12_Y, Center); 
    disp('---------- Model 07 Over ----------');
    
    % Model 08
    disp('---------- Model 08 Start ----------');
    [newcenter_08_01, sigma_08_01, W_08_01, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_08_01_X, Combine_08_01_Y, Center);
    [newcenter_08_02, sigma_08_02, W_08_02, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_08_02_X, Combine_08_02_Y, Center);
    [newcenter_08_03, sigma_08_03, W_08_03, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_08_03_X, Combine_08_03_Y, Center);
    [newcenter_08_04, sigma_08_04, W_08_04, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_08_04_X, Combine_08_04_Y, Center);
    [newcenter_08_05, sigma_08_05, W_08_05, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_08_05_X, Combine_08_05_Y, Center);
    [newcenter_08_06, sigma_08_06, W_08_06, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_08_06_X, Combine_08_06_Y, Center);
    [newcenter_08_07, sigma_08_07, W_08_07, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_08_07_X, Combine_08_07_Y, Center);
    [newcenter_08_09, sigma_08_09, W_08_09, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_08_09_X, Combine_08_09_Y, Center);
    [newcenter_08_10, sigma_08_10, W_08_10, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_08_10_X, Combine_08_10_Y, Center);
    [newcenter_08_11, sigma_08_11, W_08_11, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_08_11_X, Combine_08_11_Y, Center);
    [newcenter_08_12, sigma_08_12, W_08_12, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_08_12_X, Combine_08_12_Y, Center); 
    disp('---------- Model 08 Over ----------');
    
    % Model 09
    disp('---------- Model 09 Start ----------');
    [newcenter_09_01, sigma_09_01, W_09_01, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_09_01_X, Combine_09_01_Y, Center);
    [newcenter_09_02, sigma_09_02, W_09_02, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_09_02_X, Combine_09_02_Y, Center);
    [newcenter_09_03, sigma_09_03, W_09_03, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_09_03_X, Combine_09_03_Y, Center);
    [newcenter_09_04, sigma_09_04, W_09_04, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_09_04_X, Combine_09_04_Y, Center);
    [newcenter_09_05, sigma_09_05, W_09_05, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_09_05_X, Combine_09_05_Y, Center);
    [newcenter_09_06, sigma_09_06, W_09_06, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_09_06_X, Combine_09_06_Y, Center);
    [newcenter_09_07, sigma_09_07, W_09_07, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_09_07_X, Combine_09_07_Y, Center);
    [newcenter_09_08, sigma_09_08, W_09_08, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_09_08_X, Combine_09_08_Y, Center);
    [newcenter_09_10, sigma_09_10, W_09_10, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_09_10_X, Combine_09_10_Y, Center);
    [newcenter_09_11, sigma_09_11, W_09_11, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_09_11_X, Combine_09_11_Y, Center);
    [newcenter_09_12, sigma_09_12, W_09_12, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_09_12_X, Combine_09_12_Y, Center); 
    disp('---------- Model 09 Over ----------');
    
    % Model 10
    disp('---------- Model 10 Start ----------');
    [newcenter_10_01, sigma_10_01, W_10_01, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_10_01_X, Combine_10_01_Y, Center);
    [newcenter_10_02, sigma_10_02, W_10_02, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_10_02_X, Combine_10_02_Y, Center);
    [newcenter_10_03, sigma_10_03, W_10_03, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_10_03_X, Combine_10_03_Y, Center);
    [newcenter_10_04, sigma_10_04, W_10_04, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_10_04_X, Combine_10_04_Y, Center);
    [newcenter_10_05, sigma_10_05, W_10_05, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_10_05_X, Combine_10_05_Y, Center);
    [newcenter_10_06, sigma_10_06, W_10_06, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_10_06_X, Combine_10_06_Y, Center);
    [newcenter_10_07, sigma_10_07, W_10_07, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_10_07_X, Combine_10_07_Y, Center);
    [newcenter_10_08, sigma_10_08, W_10_08, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_10_08_X, Combine_10_08_Y, Center);
    [newcenter_10_09, sigma_10_09, W_10_09, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_10_09_X, Combine_10_09_Y, Center);
    [newcenter_10_11, sigma_10_11, W_10_11, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_10_11_X, Combine_10_11_Y, Center);
    [newcenter_10_12, sigma_10_12, W_10_12, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_10_12_X, Combine_10_12_Y, Center); 
    disp('---------- Model 10 Over ----------');
    
    % Model 11   
    disp('---------- Model 11 Start ----------');
    [newcenter_11_01, sigma_11_01, W_11_01, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_11_01_X, Combine_11_01_Y, Center);
    [newcenter_11_02, sigma_11_02, W_11_02, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_11_02_X, Combine_11_02_Y, Center);
    [newcenter_11_03, sigma_11_03, W_11_03, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_11_03_X, Combine_11_03_Y, Center);
    [newcenter_11_04, sigma_11_04, W_11_04, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_11_04_X, Combine_11_04_Y, Center);
    [newcenter_11_05, sigma_11_05, W_11_05, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_11_05_X, Combine_11_05_Y, Center);
    [newcenter_11_06, sigma_11_06, W_11_06, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_11_06_X, Combine_11_06_Y, Center);
    [newcenter_11_07, sigma_11_07, W_11_07, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_11_07_X, Combine_11_07_Y, Center);
    [newcenter_11_08, sigma_11_08, W_11_08, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_11_08_X, Combine_11_08_Y, Center);
    [newcenter_11_09, sigma_11_09, W_11_09, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_11_09_X, Combine_11_09_Y, Center);
    [newcenter_11_10, sigma_11_10, W_11_10, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_11_10_X, Combine_11_10_Y, Center);
    [newcenter_11_12, sigma_11_12, W_11_12, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_11_12_X, Combine_11_12_Y, Center); 
    disp('---------- Model 11 Over ----------');
        
    % Model 12
    disp('---------- Model 12 Start ----------');
    [newcenter_12_01, sigma_12_01, W_12_01, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_12_01_X, Combine_12_01_Y, Center);
    [newcenter_12_02, sigma_12_02, W_12_02, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_12_02_X, Combine_12_02_Y, Center);
    [newcenter_12_03, sigma_12_03, W_12_03, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_12_03_X, Combine_12_03_Y, Center);
    [newcenter_12_04, sigma_12_04, W_12_04, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_12_04_X, Combine_12_04_Y, Center);
    [newcenter_12_05, sigma_12_05, W_12_05, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_12_05_X, Combine_12_05_Y, Center);
    [newcenter_12_06, sigma_12_06, W_12_06, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_12_06_X, Combine_12_06_Y, Center);
    [newcenter_12_07, sigma_12_07, W_12_07, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_12_07_X, Combine_12_07_Y, Center);
    [newcenter_12_08, sigma_12_08, W_12_08, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_12_08_X, Combine_12_08_Y, Center);
    [newcenter_12_09, sigma_12_09, W_12_09, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_12_09_X, Combine_12_09_Y, Center);
    [newcenter_12_10, sigma_12_10, W_12_10, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_12_10_X, Combine_12_10_Y, Center);
    [newcenter_12_11, sigma_12_11, W_12_11, ~, ~] = ML2013Final_RBF_KMeans_Train(Combine_12_11_X, Combine_12_11_Y, Center); 
    disp('---------- Model 12 Over ----------');
    
    % ---------- 開始投票 ----------
    Predicted_Label = zeros(N_Test, 1);
    
    for j = 1 : N_Test
        fprintf('---------- 投給編號：%s ----------\n', num2str(j));
        Class_Vote = zeros(1, Total_Classes); 
        
        % Model 01
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_01_02, sigma_01_02, W_01_02);
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        else
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_01_03, sigma_01_03, W_01_03);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        else
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_01_04, sigma_01_04, W_01_04);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        else
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_01_05, sigma_01_05, W_01_05);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        else
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_01_06, sigma_01_06, W_01_06);	
        if (Predicted_Label_Temp(1) >= 0)        
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        else
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_01_07, sigma_01_07, W_01_07);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        else
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_01_08, sigma_01_08, W_01_08);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        else
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_01_09, sigma_01_09, W_01_09);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        else
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_01_10, sigma_01_10, W_01_10);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        else
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_01_11, sigma_01_11, W_01_11);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        else
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_01_12, sigma_01_12, W_01_12);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        else
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        end
        
        % Model 02
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_02_01, sigma_02_01, W_02_01);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        else
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_02_03, sigma_02_03, W_02_03);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        else
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_02_04, sigma_02_04, W_02_04);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        else
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_02_05, sigma_02_05, W_02_05);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        else
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_02_06, sigma_02_06, W_02_06);	
        if (Predicted_Label_Temp(1) >= 0)        
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        else
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_02_07, sigma_02_07, W_02_07);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        else
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_02_08, sigma_02_08, W_02_08);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        else
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_02_09, sigma_02_09, W_02_09);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        else
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_02_10, sigma_02_10, W_02_10);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        else
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_02_11, sigma_02_11, W_02_11);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        else
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_02_12, sigma_02_12, W_02_12);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        else
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        end
        
        % Model 03
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_03_01, sigma_03_01, W_03_01);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        else
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_03_02, sigma_03_02, W_03_02);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        else
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_03_04, sigma_03_04, W_03_04);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        else
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_03_05, sigma_03_05, W_03_05);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        else
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_03_06, sigma_03_06, W_03_06);	
        if (Predicted_Label_Temp(1) >= 0)        
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        else
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_03_07, sigma_03_07, W_03_07);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        else
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_03_08, sigma_03_08, W_03_08);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        else
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_03_09, sigma_03_09, W_03_09);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        else
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_03_10, sigma_03_10, W_03_10);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        else
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_03_11, sigma_03_11, W_03_11);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        else
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_03_12, sigma_03_12, W_03_12);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        else
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        end
        
        % Model 04        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_04_01, sigma_04_01, W_04_01);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        else
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_04_02, sigma_04_02, W_04_02);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        else
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_04_03, sigma_04_03, W_04_03);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        else
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_04_05, sigma_04_05, W_04_05);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        else
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_04_06, sigma_04_06, W_04_06);	
        if (Predicted_Label_Temp(1) >= 0)        
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        else
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_04_07, sigma_04_07, W_04_07);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        else
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_04_08, sigma_04_08, W_04_08);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        else
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_04_09, sigma_04_09, W_04_09);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        else
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_04_10, sigma_04_10, W_04_10);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        else
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_04_11, sigma_04_11, W_04_11);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        else
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_04_12, sigma_04_12, W_04_12);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        else
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        end
        
        % Model 05      
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_05_01, sigma_05_01, W_05_01);	
        if (Predicted_Label_Temp(1) >= 0)        
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        else
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_05_02, sigma_05_02, W_05_02);	
        if (Predicted_Label_Temp(1) >= 0)        
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        else
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_05_03, sigma_05_03, W_05_03);	
        if (Predicted_Label_Temp(1) >= 0)        
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        else
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_05_04, sigma_05_04, W_05_04);	
        if (Predicted_Label_Temp(1) >= 0)        
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        else
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_05_06, sigma_05_06, W_05_06);	
        if (Predicted_Label_Temp(1) >= 0)        
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        else
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_05_07, sigma_05_07, W_05_07);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        else
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_05_08, sigma_05_08, W_05_08);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        else
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_05_09, sigma_05_09, W_05_09);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        else
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_05_10, sigma_05_10, W_05_10);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        else
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_05_11, sigma_05_11, W_05_11);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        else
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_05_12, sigma_05_12, W_05_12);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        else
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        end
        
        % Model 06
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_06_01, sigma_06_01, W_06_01);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        else
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_06_02, sigma_06_02, W_06_02);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        else
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_06_03, sigma_06_03, W_06_03);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        else
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_06_04, sigma_06_04, W_06_04);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        else
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_06_05, sigma_06_05, W_06_05);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        else
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_06_07, sigma_06_07, W_06_07);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        else
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        end

        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_06_08, sigma_06_08, W_06_08);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        else
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_06_09, sigma_06_09, W_06_09);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        else
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_06_10, sigma_06_10, W_06_10);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        else
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_06_11, sigma_06_11, W_06_11);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        else
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_06_12, sigma_06_12, W_06_12);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        else
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        end
        
        % Model 07  
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_07_01, sigma_07_01, W_07_01);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        else
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_07_02, sigma_07_02, W_07_02);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        else
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_07_03, sigma_07_03, W_07_03);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        else
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_07_04, sigma_07_04, W_07_04);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        else
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_07_05, sigma_07_05, W_07_05);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        else
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_07_06, sigma_07_06, W_07_06);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        else
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_07_08, sigma_07_08, W_07_08);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        else
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_07_09, sigma_07_09, W_07_09);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        else
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_07_10, sigma_07_10, W_07_10);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        else
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_07_11, sigma_07_11, W_07_11);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        else
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_07_12, sigma_07_12, W_07_12);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        else
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        end
        
        % Model 08  
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_08_01, sigma_08_01, W_08_01);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        else
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_08_02, sigma_08_02, W_08_02);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        else
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_08_03, sigma_08_03, W_08_03);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        else
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_08_04, sigma_08_04, W_08_04);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        else
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_08_05, sigma_08_05, W_08_05);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        else
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_08_06, sigma_08_06, W_08_06);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        else
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_08_07, sigma_08_07, W_08_07);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        else
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_08_09, sigma_08_09, W_08_09);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        else
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_08_10, sigma_08_10, W_08_10);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        else
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_08_11, sigma_08_11, W_08_11);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        else
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_08_12, sigma_08_12, W_08_12);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        else
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        end
        
        % Model 09  
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_09_01, sigma_09_01, W_09_01);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        else
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_09_02, sigma_09_02, W_09_02);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        else
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_09_03, sigma_09_03, W_09_03);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        else
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_09_04, sigma_09_04, W_09_04);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        else
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_09_05, sigma_09_05, W_09_05);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        else
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_09_06, sigma_09_06, W_09_06);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        else
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_09_07, sigma_09_07, W_09_07);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        else
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_09_08, sigma_09_08, W_09_08);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        else
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_09_10, sigma_09_10, W_09_10);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        else
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_09_11, sigma_09_11, W_09_11);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        else
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_09_12, sigma_09_12, W_09_12);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        else
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        end
        
        % Model 10          
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_10_01, sigma_10_01, W_10_01);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        else
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_10_02, sigma_10_02, W_10_02);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        else
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_10_03, sigma_10_03, W_10_03);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        else
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_10_04, sigma_10_04, W_10_04);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        else
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_10_05, sigma_10_05, W_10_05);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        else
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_10_06, sigma_10_06, W_10_06);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        else
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_10_07, sigma_10_07, W_10_07);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        else
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_10_08, sigma_10_08, W_10_08);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        else
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_10_09, sigma_10_09, W_10_09);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        else
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_10_11, sigma_10_11, W_10_11);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        else
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_10_12, sigma_10_12, W_10_12);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        else
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        end
        
        % Model 11          
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_11_01, sigma_11_01, W_11_01);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        else
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_11_02, sigma_11_02, W_11_02);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        else
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_11_03, sigma_11_03, W_11_03);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        else
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_11_04, sigma_11_04, W_11_04);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        else
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_11_05, sigma_11_05, W_11_05);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        else
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_11_06, sigma_11_06, W_11_06);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        else
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_11_07, sigma_11_07, W_11_07);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        else
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_11_08, sigma_11_08, W_11_08);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        else
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_11_09, sigma_11_09, W_11_09);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        else
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_11_10, sigma_11_10, W_11_10);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        else
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_11_12, sigma_11_12, W_11_12);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        else
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        end
        
        % Model 12
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_12_01, sigma_12_01, W_12_01);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        else
            Class_Vote(1, 1) = Class_Vote(1, 1) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_12_02, sigma_12_02, W_12_02);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        else
            Class_Vote(1, 2) = Class_Vote(1, 2) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_12_03, sigma_12_03, W_12_03);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        else
            Class_Vote(1, 3) = Class_Vote(1, 3) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_12_04, sigma_12_04, W_12_04);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        else
            Class_Vote(1, 4) = Class_Vote(1, 4) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_12_05, sigma_12_05, W_12_05);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        else
            Class_Vote(1, 5) = Class_Vote(1, 5) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_12_06, sigma_12_06, W_12_06);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        else
            Class_Vote(1, 6) = Class_Vote(1, 6) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_12_07, sigma_12_07, W_12_07);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        else
            Class_Vote(1, 7) = Class_Vote(1, 7) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_12_08, sigma_12_08, W_12_08);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        else
            Class_Vote(1, 8) = Class_Vote(1, 8) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_12_09, sigma_12_09, W_12_09);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        else
            Class_Vote(1, 9) = Class_Vote(1, 9) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_12_10, sigma_12_10, W_12_10);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        else
            Class_Vote(1, 10) = Class_Vote(1, 10) + 1;
        end
        
        [Predicted_Label_Temp] = ML2013Final_RBF_KMeans_Predict(X_Test(j, :), newcenter_12_11, sigma_12_11, W_12_11);	
        if (Predicted_Label_Temp(1) >= 0)
            Class_Vote(1, 12) = Class_Vote(1, 12) + 1;
        else
            Class_Vote(1, 11) = Class_Vote(1, 11) + 1;
        end
        
        % ---------- 開票結果 ----------
        
        [Vote_Max, ~] = max(Class_Vote);
        Vote_Index = [];
        for KK = 1:Total_Classes
            if (Class_Vote(1, KK) == Vote_Max)
                Vote_Index = [Vote_Index, KK];
            end
        end        
        pp =randperm(size(Vote_Index, 2));        
        Predicted_Label(j, 1) = Vote_Index(pp(1));
        
%         [~, Predicted_Label(j, 1)] = max(Class_Vote);  
        fprintf('---------- 編號：%s 是第 %s 類----------\n', num2str(j), num2str(Predicted_Label(j, 1)));
    end
    
	% Acv_Temp  
    Acv_Temp(i, 1) = sum(Predicted_Label == Y_Test) / N_Test;
    
    % ---------- 重置 ----------
    Class_01_X = [];
    Class_02_X = [];
    Class_03_X = [];
    Class_04_X = [];
    Class_05_X = [];
    Class_06_X = [];
    Class_07_X = [];
    Class_08_X = [];
    Class_09_X = [];
    Class_10_X = [];
    Class_11_X = [];
    Class_12_X = [];

    Class_01_Y = [];
    Class_02_Y = [];
    Class_03_Y = [];
    Class_04_Y = [];
    Class_05_Y = [];
    Class_06_Y = [];
    Class_07_Y = [];
    Class_08_Y = [];
    Class_09_Y = [];
    Class_10_Y = [];
    Class_11_Y = [];
    Class_12_Y = [];
end
Acv = mean(Acv_Temp);
        
fprintf('---------- 計算結果 ----------\n');
% fprintf('Ein = %g\n', Ein);
fprintf('Acv = %g\n', Acv);

% ---------- 計時結束 ----------
toc;
