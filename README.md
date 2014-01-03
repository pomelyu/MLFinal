MLFinal
=======

2013 NTU Machine Learning Final Project


[Directory]
---------------------
```
 /data    :  raw data and processed data
 /lib     :  library we use
                * libsvm : for matlab2013 & osx10.9
                           for matlab2010 & windows 8
                * gentleboost : for multi-class adaboost 
 /log     :  log for Ein
 /result  :  result of prediction, which would be used to upload
 /save    :  temporary data or model
                * train_model : to reuse them in blending and bagging
 /src     :  source code

 MLFinal.m :  main file
```
*.dat should be put in root directory

[Usage]
---------------------
Work-flow:
In command window type 
```
MLFinal
```
Then, select appropriate training model, validation set and trainng set. If the there has been the model in /save, MLFinal would use the result immediately. Otherwise, delete the corresponding model in /save, and MLFinal would training data use the model again.

```
[1] Linear SVM with 5-fold validation in C = [1 0.1 0.01 0.001 0.0001].
[4] Gaussian SVM using downsampling with 5-fold validation in Gimma = [10 100 1000] and C = [0.1 1 10].
[7] Multi-class Adaboost

[R] to read training data. If data already in ./save/, just use it.
[T] to read test data. If data already in ./save/, just use it.
[P] Predict.
[C] Calculate Ein of training data with specific model
[E] Quit
```

[C] would calculate Ein and ask whether to record the result in ./log/
[P] predict the test data label and save the result in ./result/ for uploading

[How to add training model]
---------------------
training model is NEW_MODEL
```
% ========== Add training model chice here ==========
fprintf('==================================\n');
fprintf('-- Choose the number of Problem --\n');
fprintf('   [n*] NEW_MODEL brief\n');
% ========== End Add model choice ===================
```
In ML_Final.m, based on case'1' in switch, add 
```
% ========== Add training model here ================
case 'n'
    model_name = 'NEW_MODEL';
    model_idx  = n;
    [valid_inst, train_inst, train_label] = ChooseTrainData();
    op = ['./save/model_' model_name '_' valid_name '_' train_name '.mat'];
    % if model already exist, just load to workspace
    if exist(op, 'file') == 2
        load(op);
    else
        YOUR_PARAMETER;
        model = trainNEW_MODEL(valid_inst, train_label, train_inst, YOUR_PARAMETER);
        save(op, 'model');
    end
    clear valid_inst train_inst train_label;  
% ========== End training model ====================
```
```
% ========== Add model testing here ================
case n
    [predict_label, Eout] = testNEW_MODEL(test_label, test_inst, model);
% ========== End model testing =====================
```

In /src, add two function
```
model = trainNEW_MODEL( valid_inst, train_label, train_inst )
[predict_label, Err] = testNEW_MODEL(test_label, test_inst, model)
```
  
[Data]
---------------------
Read data by libsvmread
```
 train_raw
     train_label : N x 1      double array
     train_inst  : N x 12810  sparse matrix
                      * each image is 105 x 122 = 12810 pixel

 train_downsampling
     train_label : N x 1      double array
     train_inst  : N x 3233 sparse matric
                      * each image is 53 x 61 = 3233 pixel

 train_crop
     train_label : N x 1      double array
     train_inst  : N x 3600 sparse matric
                      * each image is 60 x 60 = 3233 pixel

```
