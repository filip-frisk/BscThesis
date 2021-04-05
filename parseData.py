from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

# Taking in TIF, normalize the Image without taking white parts into account
# Cropping 32x32 images around the center of the cells
# All images are stored in the array "images "and the labels are stored in array labels

def parseData(basePath="KI-dataset-4-types/All_Slices/", filter_name= "", label_paths="", class_names=[],set_name="No input name"):
    fileCount = len(label_paths)
    labels = []
    images = []
    print(set_name,": ")

    for path in range(fileCount):
        # Parse xml tree
        print(label_paths[path])
        tree = ET.parse(basePath+label_paths[path]+'.xml')
        root = tree.getroot()


        # Parse full image to nparray
        image = basePath+label_paths[path]+filter_name+'.tif'
        im = Image.open(image)
        imarray = np.array(im, dtype=np.double)/255
        B = imarray.copy()
        means = B.mean(axis=2)
        B[means > 0.95,:] = np.nan
        mean = np.nanmean(B, axis=(0,1))
        std = np.nanstd(B, axis=(0,1))
        #mean = np.mean(imarray[imarray != 1.0], axis = (0,1))
        #std = np.std(imarray[imarray != 1.0], axis = (0,1))
        imarray = (imarray - mean) / std

        # Loop through crops
        for child in root.iter('object'):
            # Parse label
            for name in child.iter('name'):
                label = name.text
                if label == 'w':
                    print(label_paths[path])
                    print(len(labels))
                labels.append(class_names.index(label))


            # Parse matching image data
            for box in child.iter('bndbox'):
                boundaries = []
                for val in box.iter():
                    boundaries.append(val.text)

                # Get center of crop and make sure the box fits inside of image
                meanX = int((int(boundaries[1]) + int(boundaries[3])) / 2)
                meanY = int((int(boundaries[2]) + int(boundaries[4])) / 2)

                meanX = max(meanX, 16)
                meanX = min(meanX, imarray.shape[1]-16)

                meanY = max(meanY, 16)
                meanY = min(meanY, imarray.shape[0]-16)

                cropArray = imarray[meanY-16:meanY+16, meanX-16:meanX+16, :]
                images.append(cropArray)
    return images, labels
