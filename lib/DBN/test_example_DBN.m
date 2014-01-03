%function test_example_DBN
%load('traindata.mat');
%load('testdata.mat');
load('dataset.mat');
addpath('../util');
addpath('../SAE');
addpath('../NN');
addpath('../CNN');
addpath('../CAE');



 [trainlabel]=generate_tt(trainlabel);
 [testlabel]=generate_tt(testlabel);

train_x = trainmatrix;
test_x  = testmatrix;
train_y = trainlabel';
test_y  = testlabel';

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0)
dbn.sizes = [100];
opts.numepochs =   10;
opts.batchsize = 256;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
%figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [100 100];
opts.numepochs =   10;
opts.batchsize = 256;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 12);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  10;
opts.batchsize = 256;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.10, 'Too big error');
