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
dir_out = 'D:\Classification-Products\4 - LDA Classification\Thermal\';
name = {'130606', '140416', '140606', '140829'};%'130411', '130606', '131125','140416', '140606', '140829'
iterations = 20;

for k = 1: size(name,2)
    
    %Read in Data
    speclib_table_1 = readtable(strcat(dir, cell2mat(name(k)),'_spectral_library_spectra.csv'),'ReadVariableNames',1,'Delimiter',',');
    metalib_table_1 = readtable(strcat(dir, cell2mat(name(k)),'_spectral_library_metadata.csv'),'ReadVariableNames',1,'Delimiter',',');
    speclib_table_2 = readtable(strcat(dir, '20', cell2mat(name(k)),'_MASTER_spectral_library_spectra.csv'),'ReadVariableNames',1,'Delimiter',',');
    metalib_table_2 = readtable(strcat(dir, '20', cell2mat(name(k)),'_MASTER_spectral_library_metadata.csv'),'ReadVariableNames',1,'Delimiter',',');

    % Splitting into Training/Validation
    metalib_1 = table2cell(metalib_table_1);
    speclib_1 = cell2mat(table2cell(speclib_table_1(:,6:229)));
    metalib_2 = table2cell(metalib_table_2);
    speclib_2 = cell2mat(table2cell(speclib_table_2(:,6:10)));
    remove_species = {'AGRES','ARGL','BAPI','BRNI','CECU','PISA','PLRA','POFR','PSMA','ROCK','SOIL','UMCA','URBAN'};
    for j = 1:size(remove_species,2)
        remove_index_1 = find(strcmp(remove_species(j), metalib_1(:,15))==1);
        metalib_1(remove_index_1,:) = [];
        speclib_1(remove_index_1,:) = [];
        remove_index_2 = find(strcmp(remove_species(j), metalib_2(:,15))==1);
        metalib_2(remove_index_2,:) = [];
        speclib_2(remove_index_2,:) = [];
    end

    % Remove some bad bands
    speclib_1(:,61:63) = 0;
    speclib_1(:,80:85) = 0;
    speclib_1(:,117:121) = 0;
    speclib_1(:,151:153) = 0;
    speclib_1(:,172:180) = 0;
    speclib_1(:,208:221) = 0;
    
    % Split into training and validation
    n_groups = size(unique(metalib_1(:,15)), 1);
    [name_train, name_valid] = train_val_separation(metalib_2, iterations, 0.5);
    
    table_1 = table(metalib_1(:,3),cell2mat(metalib_1(:,16)),speclib_1);
    value_1 = grpstats(table_1,'Var1','mean');
    table_2 = table(metalib_2(:,3),cell2mat(metalib_2(:,16)),speclib_2);
    value_2 = grpstats(table_2,'Var1','mean');
    index = [];
    if size(unique(metalib_1(:,3)),1) > size(unique(metalib_2(:,3)),1)
        for i = 1: size(unique(metalib_2(:,3)),1)
            index_temp = find(strcmp(table2cell(value_2(i,1)), table2cell(value_1(:,1))) == 1);
            index = vertcat(index, index_temp);
        end
        metalib = [table2cell(value_1(index,1)),table2cell(value_1(index,3))] ;
        speclib = [cell2mat(table2cell(value_1(index,4))), cell2mat(table2cell(value_2(:,4)))]; 
    elseif size(unique(metalib_1(:,3)),1) < size(unique(metalib_2(:,3)),1)
        for i = 1: size(unique(metalib_1(:,3)),1)
            index_temp = find(strcmp(table2cell(value_1(i,1)), table2cell(value_2(:,1))) == 1);
            index = vertcat(index, index_temp);
        end
        metalib = [table2cell(value_2(index,1)),table2cell(value_2(index,3))] ;
        speclib = [cell2mat(table2cell(value_1(:,4))), cell2mat(table2cell(value_2(index,4)))];
    else
        metalib = [table2cell(value_2(:,1)),table2cell(value_2(:,2))] ;
        speclib = [cell2mat(table2cell(value_1(:,4))), cell2mat(table2cell(value_2(:,4)))];    
    end

    
    overall = zeros(iterations, 1);
    kappa = zeros(iterations, 2);
    produseracc = zeros(n_groups, 2, iterations);
    errmat = zeros(n_groups+1, n_groups+1, iterations);
    cda = zeros(146, n_groups-1, iterations);
    
    for n = 1: iterations
        index_train = [];
        index_valid = [];
        
        % Pulling out spectra that are included in training polygons
        for i = 1: size(name_train,2)
            index_train_temp = find(strcmp(name_train(n,i), metalib(:,1)) == 1);
            index_train = vertcat(index_train, index_train_temp);
        end
        for i = 1: size(name_valid,2)
            index_valid_temp = find(strcmp(name_valid(n,i), metalib(:,1)) == 1);
            index_valid = vertcat(index_valid, index_valid_temp);
        end
        
        % Pulling out spectra that are included in training polygons
        speclib_train = speclib(index_train,:);
        metalib_train = cell2mat(metalib(index_train,2));
        metalib_train_poly = metalib(index_train,1);
        speclib_valid = speclib(index_valid,:);
        metalib_valid = cell2mat(metalib(index_valid,2));
        metalib_valid_poly = metalib(index_valid,1);
        
        speclib_train( :, ~any(speclib_train, 1) ) = []; % Removing band bands
        speclib_valid( :, ~any(speclib_valid, 1) ) = []; % Removing band bands
        
        % OUTPUTS:
        % accstats: accuracy stats (overall, cappa, error matrix, producers/users, valid class(what it was classified as), valid group (truth))
        % cdastats: comes out of manova1 function, eigenvectors, eigen values, all cda outputs
        
        % FUNCTION:
        [overall(n), kappa(n, :), produseracc(:, :, n), errmat(:,:,n), cda(:,:,n)] = CDA_manova(speclib_train,metalib_train,metalib_train_poly,speclib_valid,metalib_valid,metalib_valid_poly);
        
    end
    % Results
    
    mean_overall = mean(overall);
    mean_kappa = mean(kappa,1);
    mean_produser = mean(produseracc,3);
    mean_errmat = sum(errmat,3);
    mean_cda = mean(cda,3);
    group_name = unique(metalib_1(:,15));
    
    % Output CDA Results
    fileID = fopen(strcat(dir_out,cell2mat(name(k)),'_TIR_cda.csv'),'w');
    for r = 1:size(mean_cda,1)
        fprintf(fileID,'%f,',mean_cda(r,:));
        fprintf(fileID,'\n');
    end
    fclose(fileID);
    
    % Output the Classification results
    fileID = fopen(strcat(dir_out,cell2mat(name(k)),'_TIR_results.csv'),'w');
    fprintf(fileID,'%s\n', 'Overall Accuracy for all Iterations');
    fprintf(fileID,'%2.2f,', overall);
    fprintf(fileID,'\n');
    fprintf(fileID,'%s\n','Mean Overall Accuracy');
    fprintf(fileID,'%2.2f', mean_overall);
    fprintf(fileID,'\n');
    fprintf(fileID,'%s\n','Kappa for all Iterations');
    fprintf(fileID,'%.4f,',kappa(:,1));
    fprintf(fileID,'\n');
    fprintf(fileID,'%s\n',' Mean Kappa');
    fprintf(fileID,'%.4f',mean_kappa(:,1));
    fprintf(fileID,'\n');
    fprintf(fileID,'%s\n','Species, Producer Accuracy, User Accuracy');
    for r = 1:size(mean_produser,1)
        fprintf(fileID, '%s,', cell2mat(group_name(r)));
        fprintf(fileID, '%2.2f,', mean_produser(r,:));
        fprintf(fileID,'\n');
    end
    fprintf(fileID,'%s\n','Error Matrix');
    fprintf(fileID,'\n');
    for c = 1:size(group_name,1)+1
        if c == 1
            fprintf(fileID, '%s,', ' ');
        elseif c == size(group_name,1)+1
            fprintf(fileID, '%s\n', cell2mat(group_name(c-1)));
        else
            fprintf(fileID, '%s,',cell2mat(group_name(c-1)));
        end      
    end
    for r = 1:size(mean_errmat,1)-1
        fprintf(fileID,'%s,',cell2mat(group_name(r,:)));
        fprintf(fileID,'%.2f,',mean_errmat(r,:));
        fprintf(fileID,'\n');
    end
    fclose(fileID);

end
