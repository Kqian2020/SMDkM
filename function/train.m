
function model = train( J, X, Y, new_J, newX, new_Y, modelParameter)
%% parameters
lambda1    = modelParameter.lambda1;
rho        = modelParameter.rho;
maxIter    = modelParameter.maxIter;
minLoss    = modelParameter.minLoss;

[~,num_dim]  = size(X);
XTX = X'*X;
XTY = X'*Y;

%% initialization
W   = (XTX + rho*eye(num_dim)) \ (XTY); 
W_1 = W;
W_k = W;
% update C
[C] = UpdateC(J, newX, Y, new_Y);
M = C*C';
iter = 1;
oldloss = 0;
bk = 1;
bk_1 = 1;
Lip = 2*norm(XTX)^2 + 2*norm(lambda1*M)^2;
Lip  = sqrt((Lip));

while iter <= maxIter
    %% update W
    W_k  = W + (bk_1 - 1)/bk * (W - W_1);
    Wt = W_k - 1/Lip * gradientOfW(new_J,X,W,new_Y, lambda1, M);
    W_1  = W;
    W = Wt;
    bk_1 = bk;
    bk = (1 + sqrt(4*bk^2 + 1))/2;
    %% Loss
    totalloss = norm(new_J.*(X*W - new_Y),'fro')^2 + lambda1*trace(W*M*W');
    loss(iter,1) = totalloss;
    if abs((oldloss - totalloss)/oldloss) <= minLoss
        break;
    elseif totalloss <=0
        break;
    else
        oldloss = totalloss;
    end
    iter=iter+1;
end

model.W = W;
model.C = C;
model.loss = loss;
model.parameter = modelParameter;
model.iter = iter;
end

%% gradient
function gradient = gradientOfW(new_J,X,W,new_Y,lambda1, M)
gradient = X'*(new_J.*(X*W - new_Y)) + 2*lambda1*W*M;
end