'''
This Script allows generating a transistion from 1 image to another or a chain of images
'''

# Imports
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import Utils

# Main Functions
# Location Based Transistion - 2 Images
# V1 - Works only for exact pixel value matches in different locations
def I2I_Transistion_LocationBased_ExactColorMatch(I1, I2, TransistionFunc, N=5, BGColor=np.array([0, 0, 0])):
    GeneratedImgs = []

    # Get Locations of Colours in each image
    ColoursLocations_1 = ImageColourLocations(I1)
    ColoursLocations_2 = ImageColourLocations(I2)

    # V1 - Assuming Equal No of Locations of Colors in 2 Images
    # Get the Location Map
    print("Calculating Location Map...")
    LocationMap = {}
    for ck in tqdm(ColoursLocations_1.keys()):
        if ck in ColoursLocations_2.keys() and not ck == ','.join(BGColor.astype(str)):
            color = np.array(ck.split(','), int)
            # Check all possible mappings and take mapping with (customisable) movement
            mappings = list(itertools.permutations(range(len(ColoursLocations_2[ck]))))
            minError = -1
            minError_Mapping = None
            for mapping in tqdm(mappings):
                Error = 0
                for i in range(len(ColoursLocations_2[ck])):
                    Error += ((ColoursLocations_1[ck][i][0]-ColoursLocations_2[ck][mapping[i]][0])**2 + (ColoursLocations_1[ck][i][1]-ColoursLocations_2[ck][mapping[i]][1])**2)**(0.5)
                if minError == -1 or Error < minError:
                    minError = Error
                    minError_Mapping = mapping
            ChosenMapping = []
            for i in range(len(ColoursLocations_2[ck])):
                ChosenMapping.append([ColoursLocations_1[ck][i], ColoursLocations_2[ck][minError_Mapping[i]]])
            LocationMap[ck] = ChosenMapping

    # Generate Movement Transistion between Images using Custom Transistion Function
    NColorsAdded_Imgs = []
    for n in range(N):
        GeneratedImgs.append(np.ones(I1.shape, int)*BGColor)
        NColorsAdded_Imgs.append(np.zeros((I1.shape[0], I1.shape[1])).astype(int))

    # Apply Transistion for each pixel in 2 images
    print("Calculating Transistion Images...")
    for ck in tqdm(LocationMap.keys()):
        Mapping = LocationMap[ck]
        color = np.array(ck.split(','), int)
        for comb in Mapping:
            # X Movement
            X_Mov = TransistionFunc(comb[0][0], comb[1][0], N)
            # Y Movement
            Y_Mov = TransistionFunc(comb[0][1], comb[1][1], N)
            # Apply
            for n in range(N):
                if NColorsAdded_Imgs[n][X_Mov[n], Y_Mov[n]] == 0:
                    GeneratedImgs[n][X_Mov[n], Y_Mov[n]] = color
                    NColorsAdded_Imgs[n][X_Mov[n], Y_Mov[n]] = 1
                else:
                    GeneratedImgs[n][X_Mov[n], Y_Mov[n]] += color
                    NColorsAdded_Imgs[n][X_Mov[n], Y_Mov[n]] += 1
    for n in range(N):
        for i in range(NColorsAdded_Imgs[n].shape[0]):
            for j in range(NColorsAdded_Imgs[n].shape[1]):
                if NColorsAdded_Imgs[n][i, j] > 0:
                    GeneratedImgs[n][i, j] = GeneratedImgs[n][i, j] / NColorsAdded_Imgs[n][i, j]
    
    return GeneratedImgs
        
                    

# Colour Describe Function
def ImageColourLocations(I):
    ColoursLocations = {}

    if I.ndim == 2:
        I = np.reshape(I, (I.shape[0], I.shape[1], 1))
    
    for i in tqdm(range(I.shape[0])):
        for j in range(I.shape[1]):
            colourKey = ",".join(I[i, j, :].astype(str))
            if colourKey in ColoursLocations.keys():
                ColoursLocations[colourKey].append([i, j])
            else:
                ColoursLocations[colourKey] = [[i, j]]
            
    return ColoursLocations

# Driver Code
# Params
RandomImages = True

mainPath = 'TestImgs/'
imgName_1 = 'Test.png'
imgName_2 = 'Test2.png'

BGColor = [0, 0, 0]

TransistionFunc = Utils.LinearTransistion

ResizeFunc = Utils.Resize_MaxSize
ResizeParams = None

N = 50

displayDelay = 0.01

plotData = True
saveData = False

# Run Code
I1 = None
I2 = None

if not RandomImages:
    # Read Images
    I1 = cv2.cvtColor(cv2.imread(mainPath + imgName_1), cv2.COLOR_BGR2RGB)
    I2 = cv2.cvtColor(cv2.imread(mainPath + imgName_2), cv2.COLOR_BGR2RGB)

else:
    # Random Images
    Colors = [[255, 255, 0], [0, 0, 255]]
    ColorCounts = [4, 4]
    imgSize = (10, 10, 3)
    I1 = Utils.GenerateRandomImage(imgSize, BGColor, Colors, ColorCounts)
    I2 = Utils.GenerateRandomImage(imgSize, BGColor, Colors, ColorCounts)

# Resize and Show
I1, I2 = Utils.ResizeImages(I1, I2, ResizeFunc, ResizeParams)
if plotData:
    plt.subplot(1, 2, 1)
    plt.imshow(I1)
    plt.subplot(1, 2, 2)
    plt.imshow(I2)
    plt.show()

# Generate Transistion Images
GeneratedImgs = I2I_Transistion_LocationBased_ExactColorMatch(I1, I2, TransistionFunc, N, np.array(BGColor))

# Display
if plotData:
    Utils.DisplayImageSequence(GeneratedImgs, displayDelay)