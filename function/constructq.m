function [new_Y_neg, new_Y_pos,new_TE] = constructq( X,Y,Xt, Num)
[num_train,~] = size(X);
[~,num_label] = size(Y);
[num_test,~] = size(Xt);
dist_max = diag(realmax*ones(1,num_train));
temp_dist = pdist(X);
dist = squareform(temp_dist)+dist_max;
temp_dist_TE = pdist2(Xt, X);

new_Y_neg = zeros(num_train,num_label);
new_Y_pos = zeros(num_train,num_label);
%rank and weight 
temp = Num:-1:1;
my = ((repmat(temp',1,num_label))-1);

%weighted label count matrix of training set
for i = 1:num_train
    [~,index] = sort(dist(i,:));
    label_neighbor_index = index(1:Num);
    label_neighbor_neg = Y(label_neighbor_index,:);
    label_neighbor_pos = Y(label_neighbor_index,:);
    label_neighbor_neg(label_neighbor_neg == 1) = 0;
    label_neighbor_pos(label_neighbor_pos == -1) = 0;
    new_Y_neg(i,:) = sum(label_neighbor_neg .* my);
    new_Y_pos(i,:) = sum(label_neighbor_pos .* my);
end
new_Y_neg = -new_Y_neg;

%find k nearset neighbors of test
new_TE = zeros(num_test, Num);
for i = 1:num_test
    [~,index] = sort(temp_dist_TE(i,:));
    label_neighbor_index = index(1:Num);
    new_TE(i,:) = label_neighbor_index;
end
end