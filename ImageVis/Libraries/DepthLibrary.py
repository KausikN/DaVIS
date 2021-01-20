'''
Depth Functions
'''

# Imports
import cv2
import numpy as np

from Libraries.MonoDepth import calc_depth

# Main Functions
def DepthFunc_GrayScaleDepth(I):
    Depths = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    return Depths

def DepthFunc_AIDepth(I):
    # From https://github.com/nianticlabs/monodepth2

    savePath = None#"TestImgs/DepthMap.png"
    DepthsTorchArray = calc_depth.CalculateDepth(I, modelPath="ImageVis/Libraries/MonoDepth/models/mono+stereo_640x192", savePath=savePath)
    Depths = np.array(DepthsTorchArray)
    return Depths

# Driver Code