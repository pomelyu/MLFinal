MLFinal
=======

2013 NTU Machine Learning Final Project


[Directory]
---------------------
```
 /lib     :  library we use
                * libsvm : for matlab2013 & osx10.9
                           for matlab2010 & windows 8
 /log     :  log for Ein
 /result  :  result of prediction, which is used to upload
 /save    :  temporary data or model
                * train_data
                * test_data
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
In first time, Default training data would be loaded in workspace automatically. Then select appropriate training model. If the there has been the model in /save, MLFinal would use the result immediately. Otherwise, delete the corresponding model in /save, and MLFinal would training data use the model again. Finally, type [P] to predict.
Since there are no labels in test data, the prediction has no meaning.

```
[1] Linear SVM with 5-fold validation in C = [1 0.1 0.01 0.001 0.0001].
[2] Multiclass PLA
[3] MultiClass RBF-KMeans-Center
[4] Linear SVM using downsampling with 5-fold validation in C = 0.008:0.002:0.016.
[6] MultiClass RBF-OLS-Center
[7] Gaussian kernel SVM using downsampling with 5-fold validation in Gimma = [10 100 1000] and C = [0.1 1 10].
[9] MultiClass RBF-Rand-Center

[R] to read training data. If data already in ./save/, just use it.
[T] to read test data. If data already in ./save/, just use it.
[P] Predict. You shoud run training before predict.
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
    if exist('./save/model_NEW_MODEL.mat', 'file') == 2
        load ./save/model_NEW_MODEL.mat
        model_name = 'NEW_MODEL';
        model_idx = n;
    else
        if exist('train_raw_inst', 'var') == 1                    
            model = trainNEW_MODEL(train_raw_label, train_raw_inst);
            model_name = 'NEW_MODEL';
            model_idx = 1;
            save ./save/model_NEW_MODEL.mat model
        else
            fprintf('-- Please read training data\n')
        end
    end
            
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
model = trainNEW_MODEL( train_label, train_inst )
[predict_label, Err] = testNEW_MODEL(test_label, test_inst, model)
```
  
[Data]
---------------------
Read data by libsvmread
```
 train_raw_label : N x 1      double array
 train_raw_inst  : N x 12810  sparse matrix
                      * each image is 105 x 122 = 12810 pixel

 test_raw_label  : N x 1      double array
 test_raw_inst   : N x 12810  sparse matrix

 train_downsampling_inst
                 : N x 3233 sparse matric
                      * each image is 53 x 61 = 3233 pixel
```
