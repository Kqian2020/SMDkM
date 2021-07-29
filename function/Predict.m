function [TY,resluts] = Predict(modelTrain, Xt, Yt, XY, new_TE, Num)
%expand test set feature matrix
[num_test, num_L] = size(Yt');
temp = zeros(num_test,num_L);
for i=1:num_test
   temp(i,:) = sum(XY(new_TE(i,:),:))/Num; 
end
%predict and evaluate
TY = Xt*modelTrain.W;
resluts = evalt(TY',Yt);
end