function [ Result ] = evalt(Fpred,Ygnd)
 Ypred = sign(Fpred);

 %% Average Precision
AvgPrec = Average_precision(Fpred,Ygnd);
Result.AveragePrecision = AvgPrec;

%% Coverage
Cvg = coverage(Fpred,Ygnd);
Result.Coverage = Cvg;

%% One Error
OE = One_error(Fpred,Ygnd);
Result.OneError = OE;

%% Ranking Loss
RkL = Ranking_loss(Fpred,Ygnd);
Result.RankingLoss = RkL;

%% Average AUC
Result.AvgAuc = avgauc(Fpred,Ygnd);

%% Hamming Loss
Result.HammingLoss = Hamming_loss(Ypred,Ygnd);