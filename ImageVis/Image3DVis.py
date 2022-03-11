'''
Visualise Images in 3D by introducing depth to pixels based on customisable factors
'''

# Imports
import os
from turtle import width
import cv2
from matplotlib.axis import XAxis
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as graph
from PIL import Image
from tqdm import tqdm

from .Libraries import DepthLibrary
from .Libraries.VideoUtils import *
# from .Libraries import MeshLibrary

# Main Functions
def CalculateDepth(I, DepthFunc, options=None):
    Depths = np.zeros((I.shape[0], I.shape[1]))
    if I.ndim == 3:
        Depths = DepthFunc(I)

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

def ResizeImage(I, imgSize=None, keepAspectRatio=False):
    if not imgSize == None:
        size_original = [I.shape[0], I.shape[1]]
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
        # print("Original Size:", size_original)
        # print("Final Size:", I.shape)
    return I

def DisplayImage(I):
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    plt.imshow(I)
    plt.show()

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
def PlotImage3D_Plane(I, Depths, DepthLimits=None, fig=None, display=True):
    if DepthLimits is None:
        DepthLimits = [np.min(Depths)-0, np.max(Depths)+1]

    Z = Depths
    facecolors = np.array(cv2.cvtColor(I, cv2.COLOR_BGR2RGBA), dtype=np.uint8)

    I_8bit = Image.fromarray(facecolors).convert('P', palette='WEB', dither=None)
    I_idx = Image.fromarray(facecolors).convert('P', palette='WEB')
    idx_to_color = np.array(I_idx.getpalette()).reshape((-1, 3))
    colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]

    if fig is None:
        fig = graph.Figure(data=[graph.Surface(
            z=Z, surfacecolor=I_8bit, cmin=0, cmax=255, colorscale=colorscale, showscale=False
        )])
        fig.update_layout(
            title='',
            autosize=True,
            scene=dict(
                zaxis=dict(range=[0.0, 1.0])
            )
        )
    else:
        fig.update_traces(
            z=Z, surfacecolor=I_8bit
        )
    
    if display:
        plt.show()

    return fig

def PlotImage3D_Points(I, Depths, DepthLimits=(0, 1), display=True):
    if DepthLimits is None:
        DepthLimits = [np.min(Depths)-0, np.max(Depths)+1]
        
    X, Y = np.meshgrid(np.linspace(0, 1, I.shape[0]), np.linspace(0, 1, I.shape[1]))
    Z = Depths
    facecolors = cv2.cvtColor(I, cv2.COLOR_BGR2RGBA) / 255

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X, Y, Z, c=Depths)
    ax.set_zlim3d(DepthLimits[0], DepthLimits[1])
    
    if display:
        plt.show()
    
    return fig

# Driver Code
# # Params
# mainPath = 'TestImgs/'
# imgName = 'A.jpeg'
# imgSize = (250, 250)
# keepAspectRatio = True
# simplify = False

# DepthFunc = DepthLibrary.DepthFunc_AIDepth
# options = {}
# options['mods'] = ['Normalise']#, 'Reverse']
# options['NormaliseRange'] = [0, 1]
# options['DepthRange'] = [0, 255]

# DepthScale = 1
# DepthLimits = None
# ExportDepthMultiplier = np.max(np.array(imgSize))/5

# displayDepthMap = True
# display = False
# subPlots = False

# method = 2

# # Run Code
# I = cv2.imread(mainPath + imgName)
# I = ResizeImage(I, imgSize, keepAspectRatio)
# imgSize = (I.shape[0], I.shape[1])
# if display:
#     DisplayImage(I)

# print("Calculating Depths...")
# Depths = CalculateDepth(I, DepthFunc, options)
# Depths = Depths * DepthScale

# if displayDepthMap:
#     cv2.imwrite(mainPath + os.path.splitext(imgName)[0] + "_DM" + '.png', np.array((Depths*255).astype(np.uint8), dtype=np.uint8))
#     # plt.imshow(Depths, 'gray')
#     # plt.show()

# if display:
#     print("Displaying...")
#     PlotImage3D_Plane(I, Depths, DepthLimits, subPlots)

# print("Exporting...")

# # Create 3D Model METHOD 1
# if method == 1:
#     points, colors = Split3DData(I, Depths*ExportDepthMultiplier)
#     mesh = MeshLibrary.Points_to_3DModel(points, colors, method='rolling', exportPath=mainPath + os.path.splitext(imgName)[0] + '.ply', displayMesh=display)

# # Create Terrain
# if method == 2:
#     mesh = MeshLibrary.DepthImage_to_Terrain(Depths*ExportDepthMultiplier, I, mainPath + imgName, name=os.path.splitext(imgName)[0], exportPath=mainPath + os.path.splitext(imgName)[0] + '.obj')
