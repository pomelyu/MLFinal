MLFinal
=======

2013 NTU Machine Learning Final Project


[Directory]
---------------------
```
 ./src     :  source code
 ./save    :  temporary data or model
                 * train_data
                 * test_data
                 * train_model : to reuse them in blending and bagging
 ./lib     :  library we use
                 * libsvm : for matlab2013 & osx10.9
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
Default training data would be loaded in workspace automatically, and then select training model. Finally, type [P] to predict.
Since there are no labels in test data, the prediction has no meaning.

```
[1] Linear SVM without validation. Just for simple test.

[R] to read training data. If data already in ./save/, just use it.
[T] to read test data. If data already in ./save/, just use it.
[P] Predict. You shoud run training and [T] before predict.
[E] Quit
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
