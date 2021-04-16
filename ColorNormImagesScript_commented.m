% Clears all previous data
clc, clear all, close all;

% Installs the toolbox
run install.m

% Acceses the correct folder
cd 'KI-dataset-4-types'
cd 'All_Slices'

% Loads Target image
TargetImage = imread('P9_4_1.tif');
NormAll(TargetImage)
cd ..

% Name: Normall
% Function: Color Normalizes all the .tif files in the current directory
% with the three Filters: Macenkso, Reinhard and SCD
% Input: The tif file to be used as a reference.
% Output: The files are saved in the current directory.
function NormAll(TargetImage)
    Types = ["Macenko","Reinhard","SCD","RGBHist"];
    imagefiles = dir('*.tif');
    nfiles = length(imagefiles);
    for i=1:nfiles
        current_file_name = imagefiles(i).name;
        current_image = imread(current_file_name);
        for j = 1:3
            current_norm_file_name = strcat(erase(current_file_name,".tif"),"_",Types(j),".tif");
            current_norm_image = Norm(current_image, TargetImage, Types(j));
            imwrite(current_norm_image,current_norm_file_name);
            disp(current_norm_file_name)
        end
    end
end