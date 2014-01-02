addpath('./lib/libsvm');
addpath('./src');
[trainlabel, trainmatrix] = libsvmread('ml2013final_train.dat');
data.X=full(trainmatrix);
data.y=trainlabel';
model = mperceptron( data );
