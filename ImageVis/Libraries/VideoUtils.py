'''
Library for basic video functions
'''

# Imports
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .VideoInputs import *

# Main Vars
INPUTREADERS_VIDEO = {
    "Webcam": WebcamVideo,
    "Upload Video File": ReadVideo,
    "Video URL": ReadVideo_URL
}

# Main Functions
def VideoVis_Framewise(VisFunc, vid=None, path=None, max_frames=-1):
    if vid is None:
        vid = ReadVideo(path)

    fig = None
    frameCount = 0
    # Check if camera opened successfully
    if (vid.isOpened() == False): 
        print("Error opening video stream or file")
    # Read until video is completed
    while(vid.isOpened() and ((not (frameCount == max_frames)) or (max_frames == -1))):
        # Capture frame-by-frame
        ret, frame = vid.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fig = VisFunc(frame, fig=fig)
            frameCount += 1
        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    vid.release()

# Driver Code