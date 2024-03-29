'''
Algorithm Visualisation for the Dzhanibekov effect
Dzhanibekov effect Link: https://www.youtube.com/watch?v=1VPfZ_XzisU
'''

# Imports
import os
import cv2
import functools
import numpy as np
import matplotlib.pyplot as plt

from .Libraries import Plot3DVisualiseLibrary as P3L
from .Libraries import EffectFunctions

# Main Functions
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

# Image Functions
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

def DisplayImage(I):
    plt.imshow(I, 'gray')
    plt.show()

def Image2PointsColors(I, DepthFunc, ImagePointLimits):
    Depths = DepthFunc(I)

    X, Y = np.meshgrid(np.linspace(0, 1, I.shape[1]), np.linspace(0, 1, I.shape[0]))
    Z = Depths

    points = np.zeros((X.shape[0]*X.shape[1], 3))
    points[:, 0] = np.reshape(Z, (Z.shape[0]*Z.shape[1]))
    points[:, 1] = np.reshape(X, (X.shape[0]*X.shape[1]))
    points[:, 2] = np.reshape(Y, (Y.shape[0]*Y.shape[1]))[::-1]

    colors = np.reshape(I / 255, (X.shape[0]*X.shape[1], 3))

    for pli in range(len(ImagePointLimits)):
        points[:, pli] = ((points[:, pli]) * (ImagePointLimits[pli][1] - ImagePointLimits[pli][0])) + (ImagePointLimits[pli][0])

    return points, colors

# Driver Code
# # Params
# imgPath = 'TestImgs/A.jpeg'
# imgSize = (30, 30)
# keepAspectRatio = False
# DepthFunc = DepthFunc_GrayScaleDepth
# DepthOptions = {
#     'mods': ['Normalise'],#, 'Reverse']
#     'NormaliseRange': [0, 0],
#     'DepthRange': [0, 255]
#     }

# EffectFunc = functools.partial(EffectFunctions.Effect_ChaosAttractor, Func=EffectFunctions.Deriv_Lorenz)
# saveName = "IPEffect_Test"
# timeInterval = [0, 4]
# ImagePointLimits = [(-15, 15), (-15, 15), (-15, 15)]
# plotLims = [(-30, 30), (-30, 30), (0, 55)]
# speedUpFactor = 2

# frames = 250
# frame_interval = 30
# rotationSpeed = 3
# altDegrees = 30

# plotData = False
# saveData = {
#     "save": True,
#     "path":"ImageVis/GeneratedVisualisations/" + os.path.splitext(os.path.basename(imgPath))[0] + "_" + saveName + ".gif",
#     "fps": 30,
#     "figSize": [320, 240]
#     }
# # Params

# # RunCode
# DepthFunc = functools.partial(DepthFunc, options=DepthOptions)
# saveData["figSize"] = (saveData["figSize"][0]/100, saveData["figSize"][1]/100) # Change FigSize to inches (dpi = 100)
# P3L.speedUpFactor = speedUpFactor
# P3L.rotationSpeed = rotationSpeed
# P3L.altDegrees = altDegrees

# I = ReadImage(imgPath, imgSize=imgSize, keepAspectRatio=keepAspectRatio)
# DisplayImage(I)

# Points, Colors = Image2PointsColors(I, DepthFunc, ImagePointLimits)

# P3L.AnimateEffect_Generic(EffectFunc, Points, Colors, timeInterval=timeInterval, plotLims=plotLims, frames=frames, frame_interval=frame_interval, plotData=plotData, saveData=saveData)