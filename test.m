clc
clear all
addpath('./data');
%% init
eta = 0.1;
%% train
data = load('in.dta');
label = data(:,3);
label(label == -1) = 0;
data = data(:,1:2);
[w,w_b] = trainNN(data,label,1500,eta,[size(data,2),20,5,1]);
e_in = predictNN(data,label,w,w_b);

%% test
test_ = load('out.dta');
label_test = test_(:,3);
label_test(label_test == -1) = 0;
test_ = test_(:,1:2);
e_out = predictNN(test_,label_test,w,w_b);



