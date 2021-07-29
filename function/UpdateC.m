function [P] = UpdateC(J, X, Y, new_Y)
Y = J.*Y;
L = (Y*Y') - X*X';
[N, La] = size(Y);
tmpL = L - repmat(mean(L,1),N,1);
HLH = tmpL - repmat(mean(tmpL,2),1,N);
S = new_Y' * HLH * new_Y;
B = eye(La);
[P] = eig(S,B);
end