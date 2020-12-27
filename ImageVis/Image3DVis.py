'''
Visualise Images in 3D by introducing depth to pixels based on customisable factors
'''

# Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from tqdm import tqdm

import MeshLibrary


# Depth Functions
def DepthFunc_GrayScaleDepth(I, options=None):
    Depths = np.zeros((I.shape[0], I.shape[1]))
    if I.ndim == 3:
        Depths = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    if options == None:
        return Depths
    
    # Reversed GrayScale Depth
    if 'Reverse' in options['mods']:
        Depths = options['DepthRange'][1] - Depths
    # Thresholded GrayScale Depth
    if 'Threshold' in options['mods']:
        Depths[Depths < options['ThresholdRange'][0]] = options['ThresholdRange'][0]
        Depths[Depths < options['ThresholdRange'][1]] = options['ThresholdRange'][1]
    # Normalise GrapyScale Depth
    if 'Normalise' in options['mods']:
        Depths = (Depths - options['DepthRange'][0]) / options['DepthRange'][1] # Normalise to (0, 1)
        Depths = (Depths * options['NormaliseRange'][1]) + options['NormaliseRange'][0] # Normalise to custom range
    
    return Depths


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
    return I

def DisplayImage(I):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    plt.imshow(I)
    plt.show()

# Image Simplifier
def CheckColourCloseness_Dist_L1Norm(col, TopColours, minColourDist):
    col = col.split(',')
    for tc in TopColours:
        tc = tc.split(',')
        dist = 0
        for colc, tcc in zip(col, tc):
            dist += abs(int(colc) - int(tcc))
        if dist < minColourDist:
            return False
    return True

def CheckColourCloseness_Dist_L2Norm(col, TopColours, minColourDist):
    col = col.split(',')
    for tc in TopColours:
        tc = tc.split(',')
        dist = 0
        for colc, tcc in zip(col, tc):
            dist += (int(colc) - int(tcc))**2
        dist = dist**(0.5)
        if dist < minColourDist:
            return False
    return True

def EuclideanDistance(p1, p2):
    distVal = 0
    for x1, x2 in zip(p1, p2):
        distVal += (x1 - x2)**2
    return distVal ** (0.5)

def NormalisedEuclideanDistance(p1, p2):
    distVal = 0
    p1sum = np.sum(p1)
    p2sum = np.sum(p2)
    if p1sum == 0:
        p1 = [1/3]*p1.shape[0]
    else:
        p1 = p1 / np.sum(p1)
    if p2sum == 0:
        p2 = [1/3]*p2.shape[0]
    else:
        p2 = p2 / np.sum(p2)
    for x1, x2 in zip(p1, p2):
        distVal += ((x1) - (x2))**2
    return distVal ** (0.5)

def ImageColours_Describe(I):
    ColoursCount = {}
    CountColours = {}

    if I.ndim == 2:
        I = np.reshape(I, (I.shape[0], I.shape[1], 1))
    
    for i in tqdm(range(I.shape[0])):
        for j in range(I.shape[1]):
            colourKey = ",".join(I[i, j, :].astype(str))
            if colourKey in ColoursCount.keys():
                ColoursCount[colourKey] += 1
            else:
                ColoursCount[colourKey] = 1

    for colourKey in ColoursCount.keys():
        countKey = str(ColoursCount[colourKey])
        if countKey in CountColours.keys():
            CountColours[countKey].append(colourKey)
        else:
            CountColours[countKey] = [colourKey]
            
    return ColoursCount, CountColours

def StatisticalClustering_MD(I, minExtraColours=5, pixel_color_span_threshold=0.9, minColourDiff=0, DiffFunc=CheckColourCloseness_Dist_L2Norm):
    print("Describing Image Colours...")
    ColoursCount, CountColours = ImageColours_Describe(I)
    print("Unique Colours:", len(ColoursCount.keys()))
    
    # Get Top Colours
    TopColours = []
    coloursSelectedCount = 0
    pixel_span_count = 0
    threshold_pixel_span_count = int(pixel_color_span_threshold * (I.shape[0]*I.shape[1]))
    SpanHistory = []

    sortedKeys = list(CountColours.keys())
    sortedKeys.sort(key=int, reverse=True)
    TopColours.extend(CountColours[sortedKeys[0]])
    pixel_span_count += len(CountColours[sortedKeys[0]]) * int(ColoursCount[CountColours[sortedKeys[0]][0]])
    SpanHistory.append(pixel_span_count/(I.shape[0]*I.shape[1]))
    sortedKeys = sortedKeys[1:]
    for k in sortedKeys:
        if coloursSelectedCount >= minExtraColours:
            if pixel_span_count >= threshold_pixel_span_count:
                break
            for i in range(len(CountColours[k])):
                if pixel_span_count < threshold_pixel_span_count:
                    TopColours.append(CountColours[k][i])
                    pixel_span_count += int(ColoursCount[CountColours[k][i]])
                    SpanHistory.append(pixel_span_count/(I.shape[0]*I.shape[1]))
                    # print("FreqStep:", len(TopColours), pixel_span_count/(I.shape[0]*I.shape[1]), " / ", pixel_color_span_threshold)
                else:
                    break
        else:
            newCols2Add = []    
            for col in CountColours[k]:
                if DiffFunc(col, TopColours, minColourDiff):
                    newCols2Add.append(col)
            if coloursSelectedCount + len(newCols2Add) > minExtraColours:
                remainingCount = minExtraColours - coloursSelectedCount
                TopColours.extend(newCols2Add[:remainingCount])
                coloursSelectedCount = minExtraColours
                pixel_span_count += remainingCount * int(ColoursCount[CountColours[k][0]])
                # break
            else:
                TopColours.extend(newCols2Add)
                coloursSelectedCount += len(newCols2Add)
                pixel_span_count += len(newCols2Add) * int(ColoursCount[CountColours[k][0]])

    for i in range(len(TopColours)):
        TopColours[i] = np.array(TopColours[i].split(',')).astype(int)

    return TopColours, ColoursCount, CountColours

def Image_PixelClusterReplace(I, Clusters, ColoursCount, DistanceFunc=EuclideanDistance):
    # Assign Cluster Colour to each colour in colourscount
    ColourClusterMap = {}
    ReplacementsCounts = {}
    # Apply Cluster colours directly
    print("Calculating Cluster Mapping...")
    for cc in Clusters:
        ColourClusterMap[','.join(cc.astype(str))] = cc
        ReplacementsCounts[','.join(cc.astype(str))] = 0
    for ck in tqdm(ColoursCount.keys()):
        if not ck in ColourClusterMap.keys():
            # Find nearest cluster and assign
            ck_colour = np.array(list(map(int, ck.split(','))))
            minDist = None
            for tc in Clusters:
                dist = DistanceFunc(ck_colour, tc)
                if minDist == None or minDist > dist:
                    ColourClusterMap[ck] = tc
                    minDist = dist

    print("Replacing Pixel Values...")
    I_Clustered = I.copy()
    for i in tqdm(range(I_Clustered.shape[0])):
        for j in range(I_Clustered.shape[1]):
            I_Clustered[i, j] = list(ColourClusterMap[','.join(I_Clustered[i, j, :].astype(str))]).copy()
            ReplacementsCounts[','.join(I_Clustered[i, j].astype(str))] += 1

    return I_Clustered, ReplacementsCounts

def ObjectSegmentation_ColorBased(I, imgSize, minExtraColours, pixel_color_span_threshold=0.9, minColourDiff=50, DiffFunc=CheckColourCloseness_Dist_L2Norm, DistanceFunc=EuclideanDistance, CustomTopColours=None):
    # Statiscally Cluster and find Top Colours
    if CustomTopColours is not None:
        print("Describing Image Colours...")
        ColoursCount, CountColours = ImageColours_Describe(I)
        print("Unique Colours:", len(ColoursCount.keys()))
        TopColours = []
        for tc in CustomTopColours:
            TopColours.append(np.array(tc))
    else:
        TopColours, ColoursCount, CountColours = StatisticalClustering_MD(I, minExtraColours, pixel_color_span_threshold, minColourDiff, DiffFunc)
        print("N TopColors:", len(TopColours))

    # Replace pixels with closest top colour
    print("Replacing Pixels with Clusters...")
    I_Simple = None
    ReplacementCounts_Images = []
    I_r, ReplacementsCounts = Image_PixelClusterReplace(I, TopColours, ColoursCount, DistanceFunc=DistanceFunc)
    I_Simple = I_r.copy()
    ReplacementCounts_Images.append(ReplacementsCounts.copy())

    return I_Simple

# Split Functions
def Split3DData(I, Depths):
    X, Y = np.meshgrid(np.linspace(0, 1, I.shape[1]), np.linspace(0, 1, I.shape[0]))
    Z = Depths

    points = np.zeros((X.shape[0]*X.shape[1], 3))
    points[:, 0] = np.reshape(X, (X.shape[0]*X.shape[1]))
    points[:, 1] = np.reshape(Y, (Y.shape[0]*Y.shape[1]))
    points[:, 2] = np.reshape(Z, (Z.shape[0]*Z.shape[1]))

    colors = np.reshape(cv2.cvtColor(I, cv2.COLOR_BGR2RGB) / 255, (X.shape[0]*X.shape[1], 3))

    return points, colors 

# Plot Functions
def PlotImage3D_Plane(I, Depths, DepthLimits=None, subPlots=False):
    if DepthLimits is None:
        DepthLimits = [np.min(Depths)-0, np.max(Depths)+1]

    X, Y = np.meshgrid(np.linspace(0, 1, I.shape[1]), np.linspace(0, 1, I.shape[0]))
    Z = Depths
    facecolors = cv2.cvtColor(I, cv2.COLOR_BGR2RGBA) / 255

    if not subPlots:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=facecolors, shade=False)
        ax.set_zlim3d(DepthLimits[0], DepthLimits[1])
    
    else:
        ax2 = plt.subplot(1, 2, 1)
        plt.imshow(I)
        ax = plt.subplot(1, 2, 2, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=facecolors, shade=False)
        ax.set_zlim3d(DepthLimits[0], DepthLimits[1])
        
    plt.show()

def PlotImage3D_Points(I, Depths, DepthLimits=(0, 1), subPlots=False):
    if DepthLimits is None:
        DepthLimits = [np.min(Depths)-0, np.max(Depths)+1]
        
    X, Y = np.meshgrid(np.linspace(0, 1, I.shape[0]), np.linspace(0, 1, I.shape[1]))
    Z = Depths
    facecolors = cv2.cvtColor(I, cv2.COLOR_BGR2RGBA) / 255

    if not subPlots:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(X, Y, Z, c=Depths)
        ax.set_zlim3d(DepthLimits[0], DepthLimits[1])
    
    else:
        ax2 = plt.subplot(1, 2, 1)
        plt.imshow(I)
        ax = plt.subplot(1, 2, 2, projection='3d')
        ax.scatter(X, Y, Z, c=Depths)
        ax.set_zlim3d(DepthLimits[0], DepthLimits[1])
        
    plt.show()

# Driver Code
# Params
mainPath = 'TestImgs/'
imgName = 'DepthImage.png'
imgSize = (100, 100)
keepAspectRatio = True
simplify = False

DepthFunc = DepthFunc_GrayScaleDepth
options = {}
options['mods'] = ['Normalise']#, 'Reverse']
options['NormaliseRange'] = [0, 1]
options['DepthRange'] = [0, 255]

DepthScale = 1
DepthLimits = None
ExportDepthMultiplier = 25

display = False
subPlots = False

method = 2

# Run Code
I = ReadImage(mainPath + imgName, imgSize, keepAspectRatio)
imgSize = (I.shape[0], I.shape[1])
if display:
    print(imgSize)
    DisplayImage(I)

if simplify:
    minExtraColours = 1
    pixel_color_span_threshold = 0.0
    DiffFunc = CheckColourCloseness_Dist_L1Norm
    minColourDiff = 100
    DistanceFunc = EuclideanDistance
    CustomTopColours = None

    I = ObjectSegmentation_ColorBased(I, imgSize, minExtraColours, pixel_color_span_threshold=pixel_color_span_threshold, minColourDiff=minColourDiff, DiffFunc=DiffFunc, DistanceFunc=DistanceFunc, CustomTopColours=CustomTopColours)

print("Calculating Depths...")
Depths = DepthFunc(I, options)
Depths = Depths * DepthScale

if display:
    print("Displaying...")
    PlotImage3D_Plane(I, Depths, DepthLimits, subPlots)

print("Exporting...")

# Create 3D Model METHOD 1
if method == 1:
    points, colors = Split3DData(I, Depths*ExportDepthMultiplier)
    mesh = MeshLibrary.Points_to_3DModel(points, colors, method='rolling', exportPath=mainPath + os.path.splitext(imgName)[0] + '.ply', displayMesh=display)

# Create Terrain
if method == 2:
    mesh = MeshLibrary.DepthImage_to_Terrain(Depths*ExportDepthMultiplier, I, mainPath + imgName, exportPath=mainPath + os.path.splitext(imgName)[0] + '.obj')