MLFinal
=======

2013 NTU Machine Learning Final Project


[Directory]
---------------------
```
 /src     :  source code
 /save    :  temporary data or model
                * train_data
                * test_data
                * train_model : to reuse them in blending and bagging
 /lib     :  library we use
                * libsvm : for matlab2013 & osx10.9
                           for matlab2010 & windows 8
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
[4] Linear SVM using downsampling with 5-fold validation
[7] Gaussian kernel SVM using downsampling with 5-fold validation

[R] to read training data. If data already in ./save/, just use it.
[T] to read test data. If data already in ./save/, just use it.
[P] Predict. You shoud run training before predict.
[C] Calculate Ein of training data with specific model
[E] Quit
```

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
    Eout = testNEW_MODEL(test_label, test_inst, model);
% ========== End model testing =====================
```

In /src, add two function
```
model = trainNEW_MODEL( train_label, train_inst )
Err = testNEW_MODEL(test_label, test_inst, model)
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
```
