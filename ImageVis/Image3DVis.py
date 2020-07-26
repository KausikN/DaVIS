'''
Visualise Images in 3D by introducing depth to pixels based on customisable factors
'''

# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d


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
def ReadImage(imgPath, imgSize=None):
    I = cv2.imread(imgPath)
    if not imgSize == None:
        I = cv2.resize(I, imgSize)
    return I

def PlotImage3D_Plane(I, Depths, DepthLimits=(0, 1), subPlots=False):
    X, Y = np.meshgrid(np.linspace(0, 1, I.shape[0]), np.linspace(0, 1, I.shape[1]))
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
imgName = 'Test.jpg'
imgSize = (100, 100)

DepthFunc = DepthFunc_GrayScaleDepth
options = {}
options['mods'] = ['Normalise']
options['NormaliseRange'] = [0, 1]
options['DepthRange'] = [0, 255]

DepthLimits = (0, 2)

subPlots = False

# Run Code
I = ReadImage(mainPath + imgName, imgSize)
Depths = DepthFunc(I, options)
PlotImage3D_Plane(I, Depths, DepthLimits, subPlots)