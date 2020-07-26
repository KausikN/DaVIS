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

# Colour Based Gradient Transistion - 2 Images
def I2I_Transistion_ColorGradient(I1, I2, TransistionFunc, N=5):
    GeneratedImgs = []

    for n in range(N):
        GeneratedImgs.append(np.zeros(I1.shape).astype(np.uint8))

    # Apply Transistion for each pixel in 2 images
    for i in tqdm(range(I1.shape[0])):
        for j in range(I1.shape[1]):
            GeneratedPixels = TransistionFunc(I1[i, j], I2[i, j], N)
            for n in range(N):
                GeneratedImgs[n][i, j] = list(GeneratedPixels[n])

    return GeneratedImgs

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


# Driver Code
# Params
mainPath = 'TestImgs/'
imgName_1 = 'Test.jpg'
imgName_2 = 'Test2.jpg'

TransistionFunc = LinearTransistion

ResizeFunc = Resize_MaxSize
ResizeParams = None

N = 50

displayDelay = 0.01

plotData = True
saveData = False

# Run Code
# Read Images
I1 = cv2.cvtColor(cv2.imread(mainPath + imgName_1), cv2.COLOR_BGR2RGB)
I2 = cv2.cvtColor(cv2.imread(mainPath + imgName_2), cv2.COLOR_BGR2RGB)

# Resize and Show
I1, I2 = ResizeImages(I1, I2, ResizeFunc, ResizeParams)
if plotData:
    plt.subplot(1, 2, 1)
    plt.imshow(I1)
    plt.subplot(1, 2, 2)
    plt.imshow(I2)
    plt.show()

# Generate Transistion Images
GeneratedImgs = I2I_Transistion_ColorGradient(I1, I2, TransistionFunc, N)

# Display
if plotData:
    DisplayImageSequence(GeneratedImgs, displayDelay)