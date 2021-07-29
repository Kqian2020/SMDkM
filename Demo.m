%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%The MATLAB code of the paper "Self-dependence multi-label learning with double k for missing labels"
%version 1.0
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc;
addpath(genpath('.'));
%load data
load('Birds.mat');
%% parameter
modelParameter.lambda1          = 1;
modelParameter.minLoss          = 10^-3;
modelParameter.rho              = 2;
modelParameter.maxIter          = 50;
Num                             = 10;
s = RandStream.create('mrg32k3a','seed',1);
RandStream.setGlobalStream(s);

%% perpare data
data    = [train_data;test_data];
target  = double([train_target,test_target]);
[DN,~] = size(data);
[~,TN] = size(target);
%% cross validation
if(DN==TN)
    A = (1:DN)';
    cross_num = 5;
    indices = crossvalind('Kfold', A(1:DN,1), cross_num);
    All_resluts = zeros(6, cross_num);
    
    for k = 1:cross_num
        
        test = (indices == k);
        test_ID = find(test==1);
        train_ID = find(test==0);
        TE_data = data(test_ID,:);
        TR_data = data(train_ID,:);
        TE_target = target(:,test_ID);
        TR_target = target(:,train_ID)';
        
        %get missing labels
        [J] = genObv(TR_target', 0.5);
        % label recovery
        [XY, new_J, new_Y, new_TE] = doubleK(J', TR_data, TR_target, TE_data, Num);
        % train
        modelTrain  = train(J', TR_data, TR_target, new_J, [TR_data,XY], new_Y, modelParameter);
        
        %prediction and evaluation
        zz = mean(TE_target);
        TE_target(:,zz==-1) = [];
        TE_data(zz==-1,:) = [];
        [Output,resluts] = Predict(modelTrain, TE_data, TE_target, XY, new_TE, Num);
                
        All_resluts(1,k) = resluts.AveragePrecision;
        All_resluts(2,k) = resluts.AvgAuc;
        All_resluts(3,k) = resluts.HammingLoss;
        All_resluts(4,k) = resluts.Coverage;
        All_resluts(5,k) = resluts.OneError;
        All_resluts(6,k) = resluts.RankingLoss;
        
    end
    average_std = [mean(All_resluts,2) std(All_resluts,1,2)];
    PrintResults(average_std);
else
    error('Dimensional inconsistency');
end
