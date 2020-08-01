'''
This Script allows generating a transistion from 1 image to another or a chain of images
'''

# Imports
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import Utils
import ImageSimplify

# Main Functions
# V2 - Works with any 2 images - Matches Location and Color and does Transistion on both location and color
def I2I_Transistion_LocationColorBased(I1, I2, TransistionFunc_Location, TransistionParams_Location, TransistionFunc_Color, TransistionParams_Color, MappingFunc, MappingParams, N=5, BGColors=[[np.array([0, 0, 0])], [np.array([0, 0, 0])]], loadData=False):
    if not loadData:
        # Calculate Pixel Mapping
        LocationMap, ColorMap = CalculatePixelMap(I1, I2, MappingFunc, MappingParams, BGColors)

        # Save Maps
        pickle.dump(LocationMap, open(mainPath + 'LocationMap.p', 'wb'))
        pickle.dump(ColorMap, open(mainPath + 'ColorMap.p', 'wb'))
    else:
        # Load Maps
        LocationMap = pickle.load(open(mainPath + 'LocationMap.p', 'rb'))
        ColorMap = pickle.load(open(mainPath + 'ColorMap.p', 'rb'))

    # Calculate Transistion Images
    GeneratedImgs = ApplyTransistionToMapping(LocationMap, ColorMap, BGColors)
    
    return GeneratedImgs            

def CalculatePixelMap(I1, I2, MappingFunc, MappingParams, BGColors=[[np.array([0, 0, 0])], [np.array([0, 0, 0])]]):
    # Get Locations of Colours in each image
    ColoursLocations_1 = ImageColourLocations(I1)
    ColoursLocations_2 = ImageColourLocations(I2)

    Locations_1 = []
    Locations_2 = []
    Colors_1 = []
    Colors_2 = []
    for ck in ColoursLocations_1.keys():
        color = np.array(ck.split(','), int)
        if color not in BGColors[0]:
            Locations_1.extend(ColoursLocations_1[ck])
            Colors_1.extend([color]*len(ColoursLocations_1[ck]))
    for ck in ColoursLocations_2.keys():
        color = np.array(ck.split(','), int)
        if color not in BGColors[1]:
            Locations_2.extend(ColoursLocations_2[ck])
            Colors_2.extend([color]*len(ColoursLocations_2[ck]))

    # Get the Mapping
    print("Calculating Pixel Mapping...")
    print("Location Count: ", len(Locations_1), "-", len(Locations_2))
    LocationMap = []
    ColorMap = []
    LocationMap, ColorMap = MappingFunc(Locations_1, Colors_1, Locations_2, Colors_2, MappingParams)

    return LocationMap, ColorMap

def ApplyTransistionToMapping(LocationMap, ColorMap, BGColors):
    GeneratedImgs = []

    # Initialise Images and Vars
    Color_Movs = {}
    NColorsAdded_Imgs = []
    BGColor_Mov = TransistionFunc_Color(BGColors[0][0], BGColors[1][0], N)
    for n in range(N):
        GeneratedImgs.append(np.ones(I1.shape, int)*BGColor_Mov[n])
        NColorsAdded_Imgs.append(np.zeros((I1.shape[0], I1.shape[1])).astype(int))
    for cm in ColorMap:
        cmk = ','.join([','.join(np.array(cm[0]).astype(str)), ','.join(np.array(cm[1]).astype(str))])
        if cmk not in Color_Movs.keys():
            Color_Movs[cmk] = TransistionFunc_Color(cm[0], cm[1], N)

    # Apply Transistion for Mappings
    print("Calculating Transistion Images...")
    for lc, cc in tqdm(zip(LocationMap, ColorMap), disable=False):
        cmk = ','.join([','.join(np.array(cc[0]).astype(str)), ','.join(np.array(cc[1]).astype(str))])
        # Location Movement
        L_Mov = np.array(TransistionFunc_Location(lc[0], lc[1], N), int)
        X_Mov = L_Mov[:, 0]
        Y_Mov = L_Mov[:, 1]
        # Color Movement
        C_Mov = Color_Movs[cmk]
        # Apply
        for n in range(N):
            if NColorsAdded_Imgs[n][X_Mov[n], Y_Mov[n]] == 0:
                GeneratedImgs[n][X_Mov[n], Y_Mov[n]] = C_Mov[n]
                NColorsAdded_Imgs[n][X_Mov[n], Y_Mov[n]] = 1
            else:
                GeneratedImgs[n][X_Mov[n], Y_Mov[n]] += C_Mov[n]
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
RandomImages = False
SimplifyImages = True

mainPath = 'TestImgs/'
imgName_1 = 'Thresh.jpeg'
imgName_2 = 'Scenery.jpeg'

imgSize = (100, 100, 3)

BGColors = [[[0, 0, 0]], [[0, 0, 0]]]
ignoreColors_N = 1

TransistionFunc_Location = Utils.LinearTransistion
TransistionParams_Location = None
TransistionFunc_Color = Utils.LinearTransistion
TransistionParams_Color = None

MappingFunc = Utils.Mapping_LocationColorCombined
MappingParams = {'C_L_Ratio': 0.5, 'ColorSign': 1, 'LocationSign': 1}

ResizeFunc = Utils.Resize_CustomSize
ResizeParams = imgSize

N = 50
ImagePaddingCount = 5

displayDelay = 0.0001

plotData = True
saveData = True
loadData = True

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
    N_Colors_1 = 10
    ColorCount_Range_1 = (0, 50)
    N_Colors_2 = 5
    ColorCount_Range_2 = (0, 50)

    Colors_1 = list(np.random.randint(0, 255, (N_Colors_1, 3)))
    ColorCounts_1 = list(np.random.randint(ColorCount_Range_1[0], ColorCount_Range_1[1], N_Colors_1))

    Colors_2 = list(np.random.randint(0, 255, (N_Colors_2, 3)))
    ColorCounts_2 = list(np.random.randint(ColorCount_Range_2[0], ColorCount_Range_2[1], N_Colors_2))

    I1 = Utils.GenerateRandomImage(imgSize, BGColors[0], Colors_1, ColorCounts_1)
    I2 = Utils.GenerateRandomImage(imgSize, BGColors[1], Colors_2, ColorCounts_2)

    # I1 = np.zeros(imgSize, int)
    # Color1 = [255, 255, 0]
    # Color2 = [255, 0, 255]
    # I1[0, :] = Color1
    # I1[-1, :] = Color1
    # I1[:, 0] = Color1
    # I1[:, -1] = Color1
    # I2 = np.zeros(imgSize, int)
    # I2[0, :] = Color2
    # I2[-1, :] = Color2
    # I2[:, 0] = Color2
    # I2[:, -1] = Color2

# Resize
I1, I2, imgSize = Utils.ResizeImages(I1, I2, ResizeFunc, ResizeParams)

if SimplifyImages:
    # Image Color Simplification
    # Params
    maxExtraColours = 10
    minColourDiff = 100
    DiffFunc = ImageSimplify.CheckColourCloseness_Dist_L2Norm
    DistanceFunc = ImageSimplify.EuclideanDistance

    TopColors = [None, None]
    I1, TopColors[0] = ImageSimplify.ImageSimplify_ColorBased(I1, maxExtraColours, minColourDiff, DiffFunc, DistanceFunc)
    I2, TopColors[1] = ImageSimplify.ImageSimplify_ColorBased(I2, maxExtraColours, minColourDiff, DiffFunc, DistanceFunc)
    BGColors[0] = TopColors[0][:ignoreColors_N]
    BGColors[1] = TopColors[1][:ignoreColors_N]

# Show Image
if plotData:
    plt.subplot(1, 2, 1)
    plt.imshow(I1)
    plt.subplot(1, 2, 2)
    plt.imshow(I2)
    plt.show()

# Generate Transistion Images
GeneratedImgs = I2I_Transistion_LocationColorBased(I1, I2, TransistionFunc_Location, TransistionParams_Location, TransistionFunc_Color, TransistionParams_Color, MappingFunc, MappingParams, N, np.array(BGColors), loadData)
# Add Padding of I1 and I2 at ends to extend duration
for i in range(ImagePaddingCount):
    GeneratedImgs.insert(0, I1)
    GeneratedImgs.append(I2)
# Save
if saveData:
    savePath = 'TestImgs/Test.gif'
    mode = 'gif'
    frameSize = (imgSize[0], imgSize[1])
    fps = 25
    Utils.SaveImageSequence(GeneratedImgs, savePath, mode=mode, frameSize=None, fps=fps)

# Display
# if plotData:
Utils.DisplayImageSequence(GeneratedImgs, displayDelay)
