function [XY, new_J, new_Y, new_TE] = doubleK(J, TR_data, TR_target, TE_data, Num)

iter = 100;
Ymis = J .* TR_target;
%% comput label concept
% Refactor Y
[new_Y_neg, new_Y_pos, new_TE] = constructq(TR_data, J.*TR_target,TE_data, Num);
% latent semantic analysis
[P_l_q_pos, P_q_n_pos, ~] = plsa(new_Y_pos', Num, iter);
[P_l_q_neg, P_q_n_neg, ~] = plsa(new_Y_neg', Num, iter);

P_yn_pos = (P_l_q_pos * P_q_n_pos)';
P_yn_neg = (P_l_q_neg * P_q_n_neg)';

Pr_pos = P_yn_pos./(P_yn_pos + P_yn_neg);
Pr_neg = P_yn_neg./(P_yn_pos + P_yn_neg);

%% recover Z/J and Y
% missing position
temp_newY = Pr_pos;
temp_newY(temp_newY > 0.5) = 1;
temp_newY(temp_newY <= 0.5) = -1;
missing_position = temp_newY - J.*temp_newY;
% negative 
missing_position_neg = missing_position;
missing_position_neg(missing_position_neg == 1) = 0;
QQ_neg = -(missing_position_neg .* P_yn_neg);
YY_neg = missing_position_neg;
YYmis_neg_X = missing_position_neg .* Pr_neg;
% positive
missing_position_pos = missing_position;
missing_position_pos(missing_position_pos == -1) = 0;
QQ_pos = missing_position_pos .* P_yn_pos;
YY_pos = missing_position_pos;
YYmis_pos_X = missing_position_pos .* Pr_pos;

%% expand X
Ymis_neg = Ymis;
Ymis_neg(Ymis_neg == 1) = 0;
Ymis_neg_X = Ymis_neg .* Pr_neg;
Ymis_pos = Ymis;
Ymis_pos(Ymis_pos == -1) = 0;
Ymis_pos_X = Ymis_pos .* Pr_pos;

%% new J/Z and Y
new_Y = J .* TR_target + YY_neg + YY_pos;
new_J = J + QQ_neg + QQ_pos;
XY = abs(YYmis_neg_X + YYmis_pos_X + Ymis_neg_X + Ymis_pos_X);
end