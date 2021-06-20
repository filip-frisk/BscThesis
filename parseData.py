from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

# Taking in TIF, normalize the Image without taking white parts into account
# Cropping 32x32 images around the center of the cells
# All images are stored in the array "images "and the labels are stored in array labels

"""
Name: parseData
Function: Parse data from .xml and .tif file to arrays
Input: File paths and names, filter name, class names and type of dataset
Output: Cropped images of each cell and associated label (Cell type) found from input
All images are represented with a .tif-file and an associated .xml-file

Structure in .xml file from labelIMG python library used by KI:
    <object>
    <name>epithelial</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
        <xmin>1483</xmin>
        <ymin>1913</ymin>
        <xmax>1489</xmax>
        <ymax>1917</ymax>
    </bndbox>
    </object>

Each object has: Name (cell type) and bounding box
"""

def parseData(basePath="KI-dataset-4-types/All_Slices/", filter_name= "", label_paths="", class_names=[],set_name="No input name"):

    # Creates empty lists for output, calculates number of files to be loaded and prints type of dataset
    fileCount = len(label_paths)
    labels = []
    images = []
    print(set_name,": ")

    # Iterates over every file
    for path in range(fileCount):
        # Using PIL library to import files
        image = basePath+label_paths[path]+filter_name+'.tif'
        im = Image.open(image)

        # Parse full image to nparray and normalizes the image without taking into account the white background
        # Pixels with mean values over 0.95 are considered white and are ignored
        imarray = np.array(im, dtype=np.double)/255 #255 is max pixel value)
        B = imarray.copy()
        means = B.mean(axis=2)
        B[means > 0.95,:] = np.nan #Nan = ignore
        mean = np.nanmean(B, axis=(0,1))
        std = np.nanstd(B, axis=(0,1))
        imarray = (imarray - mean) / std # Normalization

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

        # Print the current file name to terminal after successful parsing
        print("Successfully loaded:", label_paths[path])

    # Remove all images in training set with labels 4 ('apoptosis / civiatte body'), not used in study
    for i in range(len(labels) - 1, -1, -1):
        if (labels[i] == 4):
            labels.pop(i)
            images.pop(i)

    return images, labels
