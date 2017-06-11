close all, clear all, clc;

N = 20000;
images = uint8(zeros(224,224,N));
gt = {};


simcep_options_randomized;
parpool(4);
parfor i = 1:N
    [im, ~, features] = simcep;
    coords = cell2mat(features.nuclei.coords);
    images(:,:,i) = im2uint8(im(:,:,3));
    gt{i} = coords;
end

save('../datasets/simcep_224px_15cells.mat', 'images', 'gt', '-v7.3');
