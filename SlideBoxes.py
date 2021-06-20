from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import random as rd
import matplotlib.pyplot as plt
import matplotlib.patches as pat

# This code generates and saves Figure 6 of the thesis.

def parseData(basePath="KI-dataset-4-types/All_Slices/", filter_name= "", label_paths="", class_names=[], colors = []):

    # Creates empty lists for output
    labels = []
    coords = []

    path = label_paths[0]
    n_crops_image = 0
    # Using PIL library to import files and transform to numpy array
    image = basePath + path + filter_name + '.tif'
    im = Image.open(image)
    imarray = np.array(im, dtype=np.uint8)

    # Create plot
    fig, ax = plt.subplots(figsize=(17, 8))
    plt.gca().set_xticks([])
    plt.xticks([])
    ax.set_xticks([])
    plt.gca().set_yticks([])
    plt.yticks([])
    ax.set_yticks([])
    ax.imshow(imarray)

    # Creates a xml tree from current file
    tree = ET.parse(basePath + path + '.xml')
    root = tree.getroot()

    # Loop through xml label tree
    for child in root.iter('object'):
        # Parse matching image data
        for box, name in zip(child.iter('bndbox'),child.iter('name')):
            label = name.text
            if class_names.index(label) == 4: contine
            # Appends corresponding label index from class_names to labels (numpy array)
            # class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial', 'epithelial', 'apoptosis / civiatte body']
            # corresponding indices = [0,1,2,3,4]
            label_idx = class_names.index(label)
            labels.append(label_idx)
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

            # Appends corresponding coordingates to coords
            # Cropping 32x32 coords around the center of the cells
            rect = pat.Rectangle((center_X-16,center_Y-16),32,32,linewidth= 1, edgecolor = colors[label_idx], facecolor = 'none')
            ax.add_patch(rect)
            #plt.plot(center_X,center_Y, 'xr')
            coords.append([center_Y-16,center_Y+16, center_X-16,center_X+16])
            n_crops_image += 1
            if n_crops_image > 200: break
    handles = [pat.Patch(color=color, label=label) for color, label in zip(colors,class_names[:-1])]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=len(handles))
    plt.savefig('SlideBoxes_'+path+'.png',dpi=300,bbox_inches = 'tight',pad_inches=0.2)
    plt.show()
    return coords, labels

if __name__ == "__main__":
    class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial', 'epithelial', 'apoptosis / civiatte body']
    colors = ['#81FF62','#61FFFF', '#000000', '#EAFF61','#F3B381', '#F3EC81']
    coords, labels = parseData(class_names = class_names, label_paths = ["P9_4_1"], colors = colors)

    

