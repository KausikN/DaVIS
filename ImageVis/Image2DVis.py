'''
Normal Image Visualisations
'''

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Main Functions
def ReadImage(imgPath, imgSize=None, keepAspectRatio=False):
    I = cv2.imread(imgPath)
    if not imgSize == None:
        size_original = [I.shape[0], I.shape[1]]
        print(size_original)
        if keepAspectRatio:
            if imgSize[1] > imgSize[0]:
                imgSize = (size_original[0] * (imgSize[1] / size_original[1]), imgSize[1])
            elif imgSize[0] > imgSize[1]:
                imgSize = (imgSize[0], size_original[1] * (imgSize[0] / size_original[0]))
            else:
                if size_original[1] > size_original[0]:
                    imgSize = (size_original[0] * (imgSize[1] / size_original[1]), imgSize[1])
                else:
                    imgSize = (imgSize[0], size_original[1] * (imgSize[0] / size_original[0]))
            imgSize = (int(round(imgSize[1])), int(round(imgSize[0])))
        I = cv2.resize(I, imgSize)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    return I

def ImageVis_RGBChannels(I):
    I_r = I * [1, 0, 0]
    I_g = I * [0, 1, 0]
    I_b = I * [0, 0, 1]

    plt.title('RGB Channels')
    plt.subplot(2, 3, 2)
    plt.imshow(I)
    plt.subplot(2, 3, 4)
    plt.imshow(I_r)
    plt.subplot(2, 3, 5)
    plt.imshow(I_g)
    plt.subplot(2, 3, 6)
    plt.imshow(I_b)
    plt.show()

    return I_r, I_g, I_b

def ImageVis_Greyscale(I):
    I_g = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    plt.title('GreyScale')
    plt.subplot(1, 2, 1)
    plt.imshow(I)
    plt.subplot(1, 2, 2)
    plt.imshow(I_g, 'gray')
    plt.show()

    return I_g

def ImageVis_DominantChannel(I, tqdmDisable=False):
    I_argmax = np.argmax(I, axis=2)
    I_max = np.max(I, axis=2)
    I_dominant = np.zeros(I.shape, dtype=np.uint8)
    for i in tqdm(range(I.shape[0]), disable=tqdmDisable):
        for j in range(I.shape[1]):
            I_dominant[i, j, I_argmax[i, j]] = I_max[i, j]
    plt.title('Dominant Channel')
    plt.subplot(1, 2, 1)
    plt.imshow(I)
    plt.subplot(1, 2, 2)
    plt.imshow(I_dominant)
    plt.show()

    return I_dominant

# Image Operations
def ImageOp_Add(I1, I2):
    I_add = np.array(I1, dtype=int) + np.array(I2, dtype=int)
    I_add = np.array(np.clip(I_add, a_min=0, a_max=255), dtype=np.uint8)

    plt.title('Add Images')
    plt.subplot(2, 2, 1)
    plt.imshow(I1, 'gray')
    plt.subplot(2, 2, 2)
    plt.imshow(I2, 'gray')
    plt.subplot(2, 2, 3)
    plt.imshow(I_add, 'gray')
    plt.show()

    return I_add

def ImageOp_Diff(I1, I2):
    I_diff = np.array(I1, dtype=int) - np.array(I2, dtype=int)
    I_diff = np.array(np.clip(I_diff, a_min=0, a_max=255), dtype=np.uint8)

    plt.title('Images Difference')
    plt.subplot(2, 2, 1)
    plt.imshow(I1, 'gray')
    plt.subplot(2, 2, 2)
    plt.imshow(I2, 'gray')
    plt.subplot(2, 2, 3)
    plt.imshow(I_diff, 'gray')
    plt.show()

    return I_diff

# Driver Code
# Params
mainPath = 'TestImgs/'
imgName = 'Pokemon2.png'

imgSize = None
keepAspectRatio = False
# Params

# RunCode
I = ReadImage(mainPath + imgName, imgSize=imgSize, keepAspectRatio=keepAspectRatio)

# Original Image
I_g = ImageVis_Greyscale(I)
ImageVis_RGBChannels(I)

# Dominant Image
I_dominant = ImageVis_DominantChannel(I)
# ImageVis_RGBChannels(I_dominant)
I_dominant_g = ImageVis_Greyscale(I_dominant)

# Differences and Adds
I_add = ImageOp_Add(I, I_dominant)
I_diff = ImageOp_Diff(I, I_dominant)

I_g_add = ImageOp_Add(I_g, I_dominant_g)
I_g_diff = ImageOp_Diff(I_g, I_dominant_g)