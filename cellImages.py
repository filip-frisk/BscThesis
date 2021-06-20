from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import random as rd

# This code parses the dataset and creates a visulisation of the cell images (Figure 7 in thesis)

def parseData(basePath="KI-dataset-4-types/All_Slices/", filter_name= "", label_paths="", class_names=[],set_name="No input name"):

    # Creates empty lists for output, calculates number of files to be loaded and prints type of dataset
    fileCount = len(label_paths)
    labels = []
    images = []

    # Iterates over every file
    for path in range(fileCount):
        n_crops_image = 0
        # Using PIL library to import files
        image = basePath+label_paths[path]+filter_name+'.tif'
        im = Image.open(image)

        # Parse full image to nparray and normalizes the image without taking into account the white background
        # Pixels with mean values over 0.95 are considered white and are ignored
        imarray = np.array(im, dtype=np.uint8)

        # Creates a xml tree from current file
        tree = ET.parse(basePath + label_paths[path] + '.xml')
        root = tree.getroot()

        # Loop through xml label tree
        for child in root.iter('object'):
            # Parse label
            for name in child.iter('name'):
                label = name.text

                # One .xml file had <name> w </name> which is handled here, which we decided to delete (N10_1_1)
                if label == 'w':
                    print(label_paths[path])
                    print(len(labels))

                # Appends corresponding label index from class_names to labels (numpy array)
                # class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial', 'epithelial', 'apoptosis / civiatte body']
                # corresponding indices = [0,1,2,3,4]
                labels.append(class_names.index(label))

            # Parse matching image data
            for box in child.iter('bndbox'):
                #Crates boundry array
                boundaries = []
                for coordinate in box.iter():
                    boundaries.append(coordinate.text)

                # Get center of crop and make sure the box fits inside of image
                center_X = int((int(boundaries[1]) + int(boundaries[3])) / 2)
                center_X = max(center_X, 16)
                center_X = min(center_X, imarray.shape[1]-16)
                center_Y = int((int(boundaries[2]) + int(boundaries[4])) / 2)
                center_Y = max(center_Y, 16)
                center_Y = min(center_Y, imarray.shape[0]-16)

                # Appends corresponding cropped to images
                # Cropping 32x32 images around the center of the cells
                cropArray = imarray[center_Y-16:center_Y+16, center_X-16:center_X+16, :]
                images.append(cropArray)
                n_crops_image += 1
                if n_crops_image > 200: break

        # Print the current file name to terminal after successful parsing
        print("Successfully loaded:", label_paths[path])

    # Remove all images in training set with labels 4 ('apoptosis / civiatte body'), not used in study
    for i in range(len(labels) - 1, -1, -1):
        if (labels[i] == 4):
            labels.pop(i)
            images.pop(i)

    return images, labels

def imageOfClassCells(cellName,images):
    height = 2
    breath = 10
    concatenate_image = []
    a = height
    for a in range(0,breath):
        start = a * height
        end = start + height
        concatenate_image.append(np.concatenate(images[start:end], axis = 0))
    plot_image = np.concatenate(concatenate_image, axis = 1)
    img = Image.fromarray(plot_image, 'RGB')
    #img.show()
    img.save(cellName+"_view.png")

if __name__ == "__main__":
    class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial', 'epithelial', 'apoptosis / civiatte body']
    images, labels = parseData(class_names = class_names, label_paths = ["P20_5_1","P9_3_1","P19_2_1"])

    for j, class_name in enumerate(class_names):
        if j > 3: continue
        c_ind = [i for i, x in enumerate(labels) if x == j]
        print(class_name,len(c_ind))
        rd.shuffle(c_ind)
        imageOfClassCells(class_name, [images[i] for i in c_ind])

