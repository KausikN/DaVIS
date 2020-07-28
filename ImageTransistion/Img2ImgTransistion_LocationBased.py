'''
This Script allows generating a transistion from 1 image to another or a chain of images
'''

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import Utils
import ImageSimplify

# Main Functions
# Location Based Transistion - 2 Images
# V1 - Works only for exact pixel value matches in different locations
def I2I_Transistion_LocationBased_ExactColorMatch(I1, I2, TransistionFunc, TransistionParams, MappingFunc, MappingParams, N=5, BGColor=np.array([0, 0, 0])):
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
            BestMapping = MappingFunc(ColoursLocations_1[ck], ColoursLocations_2[ck], MappingParams)
            LocationMap[ck] = BestMapping

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
            X_Mov = TransistionFunc(comb[0][0], comb[1][0], N, TransistionParams)
            # Y Movement
            Y_Mov = TransistionFunc(comb[0][1], comb[1][1], N, TransistionParams)
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
SimplifyImages = False

mainPath = 'TestImgs/'
imgName_1 = 'Test.png'
imgName_2 = 'Test2.png'

BGColor = [0, 0, 0]

TransistionFunc = Utils.LinearTransistion
TransistionParams = None

MappingFunc = Utils.Mapping_maxDist
MappingParams = None

ResizeFunc = Utils.Resize_MaxSize
ResizeParams = None

N = 50

displayDelay = 0.0001

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
    # Params
    imgSize = (100, 100, 3)
    N_Colors = 5
    ColorCount_Range = (0, 50)
    Colors = list(np.random.randint(0, 255, (N_Colors, 3)))
    ColorCounts = list(np.random.randint(ColorCount_Range[0], ColorCount_Range[1], N_Colors))

    I1 = Utils.GenerateRandomImage(imgSize, BGColor, Colors, ColorCounts)
    I2 = Utils.GenerateRandomImage(imgSize, BGColor, Colors, ColorCounts)

    # I1 = np.zeros(imgSize, int)
    # Color = [255, 255, 0]
    # I1[0, :] = Color
    # I1[-1, :] = Color
    # I1[:, 0] = Color
    # I1[:, -1] = Color
    # I2 = I1.copy()

if SimplifyImages:
    # Image Color Simplification
    # Params
    maxExtraColours = 5
    minColourDiff = 0
    DiffFunc = ImageSimplify.CheckColourCloseness_Dist_L2Norm
    DistanceFunc = ImageSimplify.EuclideanDistance

    I1 = ImageSimplify.ImageSimplify_ColorBased(I1, maxExtraColours, minColourDiff, DiffFunc, DistanceFunc)
    I2 = ImageSimplify.ImageSimplify_ColorBased(I2, maxExtraColours, minColourDiff, DiffFunc, DistanceFunc)

# Resize and Show
I1, I2 = Utils.ResizeImages(I1, I2, ResizeFunc, ResizeParams)
if plotData:
    plt.subplot(1, 2, 1)
    plt.imshow(I1)
    plt.subplot(1, 2, 2)
    plt.imshow(I2)
    plt.show()

# Generate Transistion Images
GeneratedImgs = I2I_Transistion_LocationBased_ExactColorMatch(I1, I2, TransistionFunc, TransistionParams, MappingFunc, MappingParams, N, np.array(BGColor))

# Display
if plotData:
    Utils.DisplayImageSequence(GeneratedImgs, displayDelay)
# Save
if saveData:
    savePath = 'TestImgs/Test.gif'
    mode = 'gif'
    frameSize = (imgSize[0], imgSize[1])
    fps = 25
    Utils.SaveImageSequence(GeneratedImgs, savePath, mode=mode, frameSize=None, fps=fps)