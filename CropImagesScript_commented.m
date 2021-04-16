% Clears all previous data
clc, clear all, close all;

% Installs the toolbox
run install.m

% Acceses the correct folder
cd 'KI-dataset-4-types'

% Loads the chosen coordinates from last use of the program
load('savedcoords')

% parameters to save images to
% 0 for false 1 for true
save_images = 0;
get_new_input = 1;
if get_new_input
    coords = containers.Map;
end

% Regions to crop pictures of
regions = ["Epithelium","Boundary","Lamina_Propria"];
files = ["All_Slices/","P9_3_1";"All_Slices/","P19_2_1";"All_Slices/","P20_5_1"];

% Iterates over the three files and run crop and then boxes
for i = 1:3
    filename = files(i,:);
    crop(filename,regions,save_images,get_new_input,coords)
    boxes(filename,regions,save_images,coords)
end

% Saves the chosen coordinates as a file
save('savedcoords','coords');
cd ..

% Class: boxes
% Function: Creates boxes highlighting the cropped parts of and image
% Input: name of the image, list of regions, parameters for saving images
% and the given coordinates of the boxes.
% Output: The files are saved in the directory.
function boxes(filename,regions,save_image,coords)
    Image_path = filename(1);
    Image_name = filename(2);
    imshow(strcat(Image_path,Image_name,'.tif'))
    colors = ['r','b','g'];
    for i = 1:3
        region_name = regions(i);
        mapkey = strcat(Image_name,region_name);
        c = coords(mapkey);
        x = c(1);
        y = c(2);
        hold on
        size = 256;
        rectangle('Position',[x,y,size,size],...
                  'EdgeColor', colors(i),...
                  'LineWidth', 3,...
                  'LineStyle','-')
    end
    if save_image
        boxes_file_name = strcat("Cropped_Image_for_Diagram_ver2/",Image_name,"_boxes_",".tif");
        f = gcf;
        exportgraphics(f,boxes_file_name,'Resolution',300);
    end
    pause(1)
    close
end

% Class: crop
% Function: Gets or uses the saved coordinates for every region we want to
% crop from and saves the cropped images.
% Input: name of the image, list of regions, parameters for saving images
% and the given coordinates of the boxes.
% Output: The files are saved in the directory.
function crop(filename,regions,save,get_new_input,coords)
    Image_path = filename(1);
    Image_name = filename(2);
    for region_name = regions
        imshow(strcat(Image_path,Image_name,'.tif'))
        title(region_name)
        mapkey = strcat(Image_name,region_name);
        if get_new_input
            [x, y, ~] = ginput(1);
            x = round(x);
            y = round(y);
            coords(mapkey) = [x,y];
        else
            c = coords(mapkey);
            x = c(1);
            y = c(2);
        end
        close
        Types = ["","_Macenko","_Reinhard","_SCD","_RGBHist"];
        size = 256;
        for j = 1:4
            current_file_name = strcat(Image_path,Image_name,Types(j),".tif");
            Image = imread(current_file_name);
            cropped_Image = imcrop(Image,[x y size size]);
            imshow(cropped_Image)
            pause(0.2)
            if save
                cropped_file_name = strcat("Cropped_Image_for_Diagram_ver2/",Image_name,Types(j),"_",region_name,"_",int2str(x),"_",int2str(y),".tif");
                imwrite(cropped_Image,cropped_file_name);
            end
        end
    end
end