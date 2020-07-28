'''
This Script allows generating a transistion from 1 image to another or a chain of images
'''

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from tqdm import tqdm

import Utils

# Main Functions
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


# Driver Code
# Params
mainPath = 'TestImgs/'
imgName_1 = 'Test.jpg'
imgName_2 = 'Test2.jpg'

TransistionFunc = Utils.LinearTransistion

ResizeFunc = Utils.Resize_MaxSize
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
I1, I2 = Utils.ResizeImages(I1, I2, ResizeFunc, ResizeParams)

I1 = Utils.GenerateGradient_LinearRadial(np.array([255, 255, 255]), np.array([255, 0, 0]), (100, 100, 3))
I2 = Utils.GenerateGradient_LinearRadial(np.array([0, 0, 255]), np.array([255, 255, 255]), (100, 100, 3))

if plotData:
    plt.subplot(1, 2, 1)
    plt.imshow(I1)
    plt.subplot(1, 2, 2)
    plt.imshow(I2)
    plt.show()

# Generate Transistion Images
GeneratedImgs = I2I_Transistion_ColorGradient(I1, I2, TransistionFunc, N)
# Loop Back to 1st image
GeneratedImgs.extend(GeneratedImgs[::-1])

# Display
if plotData:
    Utils.DisplayImageSequence(GeneratedImgs, displayDelay)