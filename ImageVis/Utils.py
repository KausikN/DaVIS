'''
This Script allows generating a transistion from 1 image to another or a chain of images
'''

# Imports
import cv2
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


            