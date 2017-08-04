%% Training/Validation
% Susan Meerdink
% 7/30/17

function [name_train, name_valid] = train_val_separation(metadata, iterations)

name_train = cell(iterations,0);
name_valid = cell(iterations,0);
dominantList = unique(metadata(:,15));

% Pull out which polygons are used for training
for d = 1:size(dominantList,1)
    name_train_poly = cell(0,0);
    name_valid_poly = cell(0,0);
    dominant_indices = strmatch(dominantList(d), metadata(:,15));
    pix_in_poly = metadata(dominant_indices,:); % Get all pixels that belong to this dominant species
    name_of_poly = unique(pix_in_poly(:,3));  % Get the number of polygons for dominant species
    num_train_poly = round(0.5 * size(name_of_poly, 1));  % Determine how many polygons are going to be set aside for training
    for t = 1:iterations
        idx_valid = (1:size(name_of_poly,1))';
        [name_train_poly(t,:), idx] = datasample(name_of_poly, num_train_poly,'Replace',false);
        idx_valid(sort(idx)) = [];
        name_valid_poly(t,:) = name_of_poly(idx_valid);
    end
    name_train = [name_train name_train_poly];
    name_valid = [name_valid name_valid_poly];
end

end
