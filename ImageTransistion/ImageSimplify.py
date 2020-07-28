'''
Object Segmentation using Colour Clustering
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os

saveExt = '.png'

# Main Functions
# Distance Functions
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

# Check Functions
def CheckColourCloseness_Diff(col, TopColours, minColourDiff):
    col = col.split(',')
    for tc in TopColours:
        tc = tc.split(',')
        for colc, tcc, minc in zip(col, tc, minColourDiff):
            if abs(int(colc) - int(tcc)) < minc:
                return False
    return True

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

# Statistical Clustering
# Statistical Clustering with min Color Diff Threshold
def StatisticalClustering_MD(I, maxExtraColours=5, minColourDiff=0, DiffFunc=CheckColourCloseness_Dist_L1Norm):
    print("Describing Image Colours...")
    ColoursCount, CountColours = ImageColours_Describe(I)
    print("Unique Colours:", len(ColoursCount.keys()))
    
    # Get Top Colours
    TopColours = []
    coloursSelectedCount = 0
    sortedKeys = list(CountColours.keys())
    sortedKeys.sort(key=int, reverse=True)
    TopColours.extend(CountColours[sortedKeys[0]])
    sortedKeys = sortedKeys[1:]
    for k in sortedKeys:
        newCols2Add = []
        for col in CountColours[k]:
            if DiffFunc(col, TopColours, minColourDiff):
                newCols2Add.append(col)
        if coloursSelectedCount + len(newCols2Add) > maxExtraColours:
            remainingCount = maxExtraColours - coloursSelectedCount
            TopColours.extend(newCols2Add[:remainingCount])
            coloursSelectedCount = maxExtraColours
            break
        else:
            TopColours.extend(newCols2Add)
            coloursSelectedCount += len(newCols2Add)

    for i in range(len(TopColours)):
        TopColours[i] = np.array(TopColours[i].split(',')).astype(int)

    return TopColours, ColoursCount, CountColours

# Image Functions
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

# Simplify Functions
def ImageSimplify_ColorBased(I, maxExtraColours, minColourDiff=50, DiffFunc=CheckColourCloseness_Dist_L1Norm, DistanceFunc=EuclideanDistance):
    DistanceFuncName = DistanceFunc.__name__

    # Statiscally Cluster and find Top Colours
    TopColours, ColoursCount, CountColours = StatisticalClustering_MD(I, maxExtraColours, minColourDiff, DiffFunc)

    # Replace pixels with closest top colour
    print("Replacing Pixels with Clusters...")
    I_r, ReplacementsCounts = Image_PixelClusterReplace(I, TopColours, ColoursCount, DistanceFunc=DistanceFunc)
    
    return I_r, TopColours