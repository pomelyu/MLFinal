function model=trainDBN(train_label, train_inst)
%load('traindata.mat');
%load('testdata.mat');
%load('dataset.mat');
addpath('./lib/util');
addpath('./lib/SAE');
addpath('./lib/NN');
addpath('./lib/CNN');
addpath('./lib/CAE');
addpath('./lib/DBN');



 [train_label]=generate_tt(train_label);

train_x = train_inst;
train_y = train_label';
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
opts.batchsize = 64;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
model = dbnunfoldtonn(dbn, 12);
model.activation_function = 'sigm';

%train nn
opts.numepochs =  10;
opts.batchsize = 64;
model = nntrain(model, train_x, train_y, opts);
