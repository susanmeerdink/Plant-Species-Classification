%% Plant Species Classification
% Susan Meerdink
% 7/22/17
% This script is the main function that will load the spectral libraries
% (formed using python script) and split them into training/validation
% iteratively for the various dimension reduction and classification
% techniques. Formly I was using CDA/LDA for classification.

%  Dimensionality reduction 
% 	1. Kernel MNF
% 	2. Kernel PCA
%   3. Kernel CDA
% Classifiers:
% 	1. Artificial Neural Networks with Manifold for uncertainty analysis (NEEDS DR)
% 	2. Gaussian Processing Regression (can have DR or no DR)
% 	3. Kernel Canonical Discriminant Analysis (can have DR or no DR) uses mediods
%% Load data
dir = 'D:\Classification-Products\1 - Spectral Library\Combined Single Date\';
name_140416 = '140416_spectral_library';

speclib_140416 = readtable(strcat(dir, name_140416,'_spectra.csv'),'ReadVariableNames',1);
metalib_140416 = readtable(strcat(dir, name_140416,'_metadata.csv'),'ReadVariableNames',1);

%% Splitting into Training/Validation
metalib_all = table2cell(metalib_140416);
name_train_polys = [];
dominantList = unique(metalib_all(:,14));

% Pull out which polygons are used for training
for d = 1:size(dominantList,1)
    pix_in_poly = metalib_all(strmatch(dominantList(d), metalib_all(:,14)),:); % Get all pixels that belong to this dominant species
    name_of_poly = unique(pix_in_poly(:,3));  % Get the number of polygons for dominant species
    num_train_poly = round(0.2 * size(name_of_poly, 1));  % Determine how many polygons are going to be set aside for training
    name_train_poly = cell(10, num_train_poly);  % Array to hold index of trained polygons 
    for t = 1:10
        name_train_poly(t,:) = datasample(name_of_poly, num_train_poly);
    end
    name_train_polys = [name_train_polys name_train_poly];    
end

%% Dimensionality Reduction
% Currently CDA

speclib_all = cell2mat(table2cell(speclib_140416(:,6:229)));
speclib_train = [];
metalib_train =[];
speclib_valid = [];
metalib_valid = [];

% Pulling out spectra that are included in training polygons
for i = 1: size(name_train_polys,2)
    index_train = find(strcmp(name_train_polys(1,i), metalib_all(:,3)) == 1);
    index_valid = find(strcmp(name_train_polys(1,i), metalib_all(:,3)) == 0);
    speclib_train = vertcat(speclib_train, speclib_all(index_train,:));
    metalib_train = vertcat(metalib_train, metalib_all(index_train,:));
    speclib_valid = vertcat(speclib_valid, speclib_all(index_valid,:));
    metalib_valid = vertcat(metalib_valid, metalib_all(index_valid,:));
end

speclib_train( :, ~any(speclib_train,1) ) = []; % Removing band bands
speclib_valid( :, ~any(speclib_valid,1) ) = []; % Removing band bands

% Run manova for training library   
[d, p, cdastats] = manova1(speclib_train, metalib_train(:,14));
n_groups = size(unique(metalib_train(:,14)), 1);

% Calculate canonical variables for training and validation libraries
% n=nspec, m=nbands, p=ngroups-1
canon_vars_Train = speclib_train * cdastats.eigenvec(:, 1:n_groups-1);  %(n x m) * (m x p) = (n x p)
canon_vars_Valid = speclib_valid * cdastats.eigenvec(:, 1:n_groups-1);  %(n x m) * (m x p) = (n x p)

%% Classification
% Currently using LDA

valid_class = classify(canon_vars_Valid, canon_vars_Train, metalib_train(:,14));

%% Classification Accuracy
%Overall Accuracy
num_valid = length(metalib_valid(:,14));
correct = strcmp(valid_class, metalib_valid(:,14));
overall_acc = 100*(1 - (num_valid - sum(correct)) / num_valid);

%Error Matrix
%in error matrix, rows (j) are classified class and cols (i) are true class
errmat = zeros(n_groups+1, n_groups+1);
groupIDs = unique(metalib_valid(:,14));
unmod_value = 9999;
groupIDs_wunmod = vertcat(groupIDs, unmod_value);
for j = 1:n_groups+1
    for i = 1:n_groups+1
        temp1 = find(strcmp(valid_class, groupIDs_wunmod(j)));
        temp2 = find(strcmp(metalib_valid(:,14), groupIDs_wunmod(i)));
        temp = intersect(temp1,temp2);
        count = length(temp);
        errmat(j,i)=count;
    end
end

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
accstats.overall=overall_acc;
accstats.kappa = [kappa;kappa_var];
accstats.prodacc = prod_acc(:);
accstats.useracc = user_acc(:);
accstats.errmat=errmat;
