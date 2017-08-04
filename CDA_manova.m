function [overall, kappa, produseracc, errmat, cda] = ...
            CDA_manova(trainlib,train_group,train_poly,validlib,valid_pixel_group,valid_poly)


%matrix multiplication of original data by canonical coeffs to create the
%actual canonical variables as linear comb of original vars

% Execute on Polygon level
table_train = table(train_poly,train_group,trainlib);
values_train = grpstats(table_train,'train_poly','mean');
train_group = cell2mat(table2cell(values_train(:,3)));
trainlib = cell2mat(table2cell(values_train(:,4:end)));

% Put results in table to get polygon level classification
table_valid = table(valid_poly,valid_pixel_group,validlib);
values_valid = grpstats(table_valid,'valid_poly','mean');
valid_group = cell2mat(table2cell(values_valid(:,3)));
validlib = cell2mat(table2cell(values_valid(:,4:end)));

%Run manova for training library
[d,p,cdastats] = manova1(trainlib,train_group);
n_groups=length(unique(train_group));

%Calculate canonical variables for training and validation libraries
%n=nspec, m=nbands, p=ngroups-1
canon_vars_Train = trainlib*cdastats.eigenvec(:,1:n_groups-1);  %(n x m) * (m x p) = (n x p)
canon_vars_Valid = validlib*cdastats.eigenvec(:,1:n_groups-1);  %(n x m) * (m x p) = (n x p)

%use LDA to classify
valid_class = classify(canon_vars_Valid,canon_vars_Train,train_group);

%% Calculate Accuracy Metrics
%Overall Accuracy
nspec_valid=length(valid_group);
correct = valid_class==valid_group; % strcmp(valid_class, valid_group);
overall_acc = 100*(1-((nspec_valid - sum(correct)) / nspec_valid));

%Error Matrix
%in error matrix, rows (j) are classified class and cols (i) are true class
errmat = zeros(n_groups+1,n_groups+1);
groupIDs = unique(valid_group);
unmod_value = 9999;
groupIDs_wunmod = vertcat(groupIDs,unmod_value);
for j=1:n_groups+1
    for i=1:n_groups+1
        temp1=find(valid_class == groupIDs_wunmod(j));
        temp2=find(valid_group == groupIDs_wunmod(i));
        temp=intersect(temp1,temp2);
        count=length(temp);
        errmat(j,i)=count;
    end
end
% for j=1:n_groups+1
%     for i=1:n_groups+1
%         temp1=find(strcmp(valid_class, groupIDs_wunmod(j)));
%         temp2=find(strcmp(valid_group, groupIDs_wunmod(i)));
%         temp=intersect(temp1,temp2);
%         count=length(temp);
%         errmat(j,i)=count;
%     end
% end

%Kappa
theta1=zeros(n_groups,1);
theta2=zeros(n_groups,1);
theta3=zeros(n_groups,1);
theta4=zeros(n_groups,n_groups);
matrix_sumA=sum(errmat);
matrix_sum=sum(matrix_sumA);
for i=1:n_groups
    theta1(i)=errmat(i,i)/matrix_sum;
    theta2(i)=sum(errmat(:,i))*sum(errmat(i,:))/(matrix_sum^2);
    theta3(i)=errmat(i,i)*(sum(errmat(:,i))+sum(errmat(i,:)))/(matrix_sum^2);
    for j=1:n_groups
        theta4(j,i)=errmat(j,i)*((sum(errmat(:,i))+sum(errmat(j,:)))^2)/(matrix_sum^3);
    end
end
theta1_sum=sum(theta1);
theta2_sum=sum(theta2);
theta3_sum=sum(theta3);
theta4_sumA=sum(theta4);
theta4_sum=sum(theta4_sumA);
kappa=(theta1_sum-theta2_sum)/(1-theta2_sum);

%Kappa Variance
var1=(1/matrix_sum);
var2=((theta1_sum*(1-theta1_sum))/((1-theta2_sum)^2));
var3=((2*(1-theta1_sum)*(2*theta1_sum*theta2_sum-theta3_sum))/((1-theta2_sum)^3));
var4=((((1-theta1_sum)^3)*(theta4_sum-4*(theta2_sum^2)))/((1-theta2_sum)^4));
kappa_var=var1*(var2+var3+var4);

%Producer's & User's Accuracies
for i=1:n_groups
    prod_acc(i)=errmat(i,i)/sum(errmat(:,i))*100; %sum by col
    user_acc(i)=errmat(i,i)/sum(errmat(i,:))*100; %sum by row
end
% Output Accuracy Stats
%     accstats.overall=overall_acc;
%     accstats.kappa=[kappa;kappa_var];
%     accstats.produseracc=horzcat(prod_acc(:),user_acc(:));
%     accstats.errmat=errmat;
overall = overall_acc;
kappa = [kappa;kappa_var];
produseracc = horzcat(prod_acc(:),user_acc(:));
cda = cdastats.eigenvec(:,1:n_groups-1);
return