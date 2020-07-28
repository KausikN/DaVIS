'''
This Script allows generating a transistion from 1 image to another or a chain of images
'''

# Imports
import cv2
import random
import imageio
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from tqdm import tqdm

# Main Functions
def ResizeImages(I1, I2, ResizeFunc=None, ResizeParams=None):
    print("Before Resizing: I1:", I1.shape, "I2:", I2.shape)
    # Resize Match the 2 images - Final I1 and I2 must be of same size
    if not ResizeFunc == None:
        if ResizeParams == None:
            I1, I2 = ResizeFunc(I1, I2)
        else:
            I1, I2 = ResizeFunc(I1, I2, ResizeParams)
    print("After Resizing: I1:", I1.shape, "I2:", I2.shape)
    return I1, I2

def DisplayImageSequence(ImgSeq, delay=1):
    imgIndex = 0
    N = len(ImgSeq)
    while(True):
        plt.figure(1)
        plt.clf()
        plt.imshow(ImgSeq[imgIndex])
        plt.title(str(imgIndex+1))

        plt.pause(delay)
        imgIndex = (imgIndex + 1) % N

def SaveImageSequence(ImgSeq, path, mode='gif', frameSize=None, fps=25):
    # modes
    # gif
    if mode in ['gif', 'GIF', 'G', 'g']:
        imageio.mimsave(path, ImgSeq)
    # Video
    elif mode in ['V', 'v', 'Video', 'video', 'VIDEO', 'VID', 'vid']:
        if frameSize == None:
            frameSize = (ImgSeq[0].shape[0], ImgSeq[0].shape[1])
        vid = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), fps, frameSize)
        for i in range(len(ImgSeq)):
            vid.write(ImgSeq[i])
        vid.release()
    # Images
    else:
        for i in range(len(ImgSeq)):
            cv2.imwrite(path + str(i+1), ImgSeq[i])


# Mapping Functions
# Mapping Functions
def Mapping_BruteForce(L1, L2):
    # Check all possible mappings and take mapping with (customisable) movement
    mappings = list(itertools.permutations(range(len(L2))))
    minError = -1
    minError_Mapping = None
    for mapping in tqdm(mappings):
        Error = 0
        for i in range(len(L2)):
            Error += ((L1[i][0]-L2[mapping[i]][0])**2 + (L1[i][1]-L2[mapping[i]][1])**2)**(0.5)
        if minError == -1 or Error < minError:
            minError = Error
            minError_Mapping = mapping

    ChosenMapping = []
    for i in range(len(L2)):
        ChosenMapping.append([L1[i], L2[minError_Mapping[i]]])
        
    return ChosenMapping

def Mapping_minDist(L1, L2):
    minDist_Mapping = []
    for p1 in L1:
        minDist = -1
        minDist_Point = -1
        for p2 in L2:
            dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**(0.5)
            if minDist == -1 or dist < minDist:
                minDist = dist
                minDist_Point = p2.copy()
        minDist_Mapping.append([p1, minDist_Point])
        L2.remove(minDist_Point)
    return minDist_Mapping

def Mapping_maxDist(L1, L2):
    maxDist_Mapping = []
    for p1 in L1:
        maxDist = -1
        maxDist_Point = -1
        for p2 in L2:
            dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**(0.5)
            if maxDist == -1 or dist > maxDist:
                maxDist = dist
                maxDist_Point = p2.copy()
        maxDist_Mapping.append([p1, maxDist_Point])
        L2.remove(maxDist_Point)
    return maxDist_Mapping

# Transistion Functions
def LinearTransistion(c1, c2, N):
    C_Gen = None
    # If colours
    if type(c1) == type([]):
        C_Gen = np.zeros((N, len(c1)))
        for i in range(len(c1)):
            C_Gen[:, i] = np.linspace(c1[i], c2[i], N).astype(np.uint8)
    else:
        C_Gen = np.linspace(c1, c2, N).astype(np.uint8)
    return list(C_Gen)

# Resize Functions
def Resize_CustomSize(I1, I2, Size):
    I1 = cv2.resize(I1, Size)
    I2 = cv2.resize(I2, Size)
    return I1, I2

def Resize_MaxSize(I1, I2):
    CommonSize = (max(I1.shape[0], I2.shape[0]), max(I1.shape[1], I2.shape[1]))
    I1 = cv2.resize(I1, CommonSize)
    I2 = cv2.resize(I2, CommonSize)
    return I1, I2

def Resize_PaddingFillMaxSize(I1, I2):
    # Colour Images
    if I1.ndim == 3:
        # Find Max Size and create new images
        CommonSize = (max(I1.shape[0], I2.shape[0]), max(I1.shape[1], I2.shape[1]), I1.shape[2])
    else:
        # Find Max Size and create new images
        CommonSize = (max(I1.shape[0], I2.shape[0]), max(I1.shape[1], I2.shape[1]))

    I1_R = np.zeros(CommonSize).astype(int)
    I2_R = np.zeros(CommonSize).astype(int)
    # Fill Image Parts and align to center
    # I1
    PaddingSize = (CommonSize[0] - I1.shape[0], CommonSize[1] - I1.shape[1])
    ImgPart_Start = (int(PaddingSize[0]/2), int(PaddingSize[1]/2))
    ImgPart_End = (ImgPart_Start[0] + I1.shape[0], ImgPart_Start[1] + I1.shape[1])
    I1_R[ImgPart_Start[0]:ImgPart_End[0], ImgPart_Start[1]:ImgPart_End[1]] = I1
    # I2
    PaddingSize = (CommonSize[0] - I2.shape[0], CommonSize[1] - I2.shape[1])
    ImgPart_Start = (int(PaddingSize[0]/2), int(PaddingSize[1]/2))
    ImgPart_End = (ImgPart_Start[0] + I2.shape[0], ImgPart_Start[1] + I2.shape[1])
    I2_R[ImgPart_Start[0]:ImgPart_End[0], ImgPart_Start[1]:ImgPart_End[1]] = I2

    return I1_R, I2_R

# Image Generation Functions
def GenerateGradient_LinearRadial(innerColor, outerColor, imgSize):
    centerPixel = (int(imgSize[0]/2), int(imgSize[1]/2))

    I = np.zeros(imgSize).astype(np.uint8)
    I[centerPixel[0], centerPixel[1]] = innerColor
    I[-1, -1] = outerColor # Outer most pixel in any case of size is final pixel
    maxDist = ((imgSize[0]-centerPixel[0])**2 + (imgSize[1]-centerPixel[1])**2)**(0.5)

    # Color Images
    if len(imgSize) <= 2:
        for i in range(imgSize[0]):
            for j in range(imgSize[1]):
                dist = ((i-centerPixel[0])**2 + (j-centerPixel[1])**2)**(0.5)
                fracVal = dist / maxDist
                I[i, j] = int(outerColor * fracVal + innerColor * (1-fracVal))
    # Grayscale Images
    else:
        for i in range(imgSize[0]):
            for j in range(imgSize[1]):
                dist = ((i-centerPixel[0])**2 + (j-centerPixel[1])**2)**(0.5)
                fracVal = dist / maxDist
                I[i, j] = list((outerColor * fracVal + innerColor * (1-fracVal)).astype(np.uint8))

    return I


def GenerateRandomImage(imgSize, BGColor, Colors, ColorCounts):
    I = np.ones(imgSize, int)*BGColor
    totalPixelCount = imgSize[0]*imgSize[1]
    colorPixelsCount = sum(ColorCounts)
    BGColorCount = totalPixelCount - colorPixelsCount
    if BGColorCount >= 0:
        order = np.array([-1]*totalPixelCount)
        curIndex = 0
        for i in range(len(ColorCounts)):
            order[curIndex : curIndex + ColorCounts[i]] = i
            curIndex += ColorCounts[i]
        random.shuffle(order)
    I_Colors = np.reshape(np.array(order), (imgSize[0], imgSize[1]))
    for i in range(I_Colors.shape[0]):
        for j in range(I_Colors.shape[1]):
            if not I_Colors[i, j] == -1:
                I[i, j] = Colors[I_Colors[i, j]]
    return I