"""
Stream lit GUI for hosting DaVIS
"""

# Imports
from opcode import stack_effect
import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import json

from ImageVis import Image2DVis
from ImageVis import Image3DVis

# Main Vars
config = json.load(open('./StreamLitGUI/UIConfig.json', 'r'))

# Main Functions
def main():
    # Create Sidebar
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
        tuple(
            [config['PROJECT_NAME']] + 
            config['PROJECT_MODES']
        )
    )
    
    if selected_box == config['PROJECT_NAME']:
        HomePage()
    else:
        correspondingFuncName = selected_box.replace(' ', '_').lower()
        if correspondingFuncName in globals().keys():
            globals()[correspondingFuncName]()
 

def HomePage():
    st.title(config['PROJECT_NAME'])
    st.markdown('Github Repo: ' + "[" + config['PROJECT_LINK'] + "](" + config['PROJECT_LINK'] + ")")
    st.markdown(config['PROJECT_DESC'])

    # st.write(open(config['PROJECT_README'], 'r').read())

#############################################################################################################################
# Repo Based Vars
CACHE_PATH = "StreamLitGUI/CacheData/Cache.json"
DEFAULT_IMAGE_PATH = "TestImgs/Arch.jpeg"

# Util Vars
CACHE = {}

# Util Functions
def LoadCache():
    global CACHE
    CACHE = json.load(open(CACHE_PATH, 'r'))

def SaveCache():
    global CACHE
    json.dump(CACHE, open(CACHE_PATH, 'w'), indent=4)

# Main Functions


# UI Functions
def UI_LoadImage():
    USERINPUT_ImageData = st.file_uploader("Upload Image", ['png', 'jpg', 'jpeg', 'bmp'])
    if USERINPUT_ImageData is not None:
        USERINPUT_ImageData = USERINPUT_ImageData.read()
    else:
        USERINPUT_ImageData = open(DEFAULT_IMAGE_PATH, 'rb').read()
    USERINPUT_ImageData = cv2.imdecode(np.frombuffer(USERINPUT_ImageData, np.uint8), cv2.IMREAD_COLOR)
    USERINPUT_Image = cv2.cvtColor(USERINPUT_ImageData, cv2.COLOR_BGR2RGB)
    return USERINPUT_Image

# Repo Based Functions
def image_2d_vis():
    # Title
    st.header("Image 2D Vis")

    # Prereq Loaders

    # Load Inputs
    USERINPUT_Image = UI_LoadImage()

    # Process Inputs
    # RGB Gray
    I_r, I_g, I_b = Image2DVis.ImageVis_RGBChannels(USERINPUT_Image, display=False)
    I_gray = Image2DVis.ImageVis_Greyscale(USERINPUT_Image, display=False)
    # Dominant Lowest Channels
    I_dom = Image2DVis.ImageVis_DominantChannel(USERINPUT_Image, display=False)
    I_low = Image2DVis.ImageVis_LowestChannel(USERINPUT_Image, display=False)
    # Histogram Enhance
    histData = Image2DVis.ImageHistogram_2D(USERINPUT_Image, bins=list(range(0, 256)), display=False)
    I_histenhanced = Image2DVis.ImageHistogram_Enhance(USERINPUT_Image, histData, selectRange=[0, 255], display=False)

    # Display Outputs
    # Original
    st.image(USERINPUT_Image, caption="Original", use_column_width=True)
    # RGB Gray
    col1, col2, col3 = st.columns(3)
    col1.image(I_r, caption="Red Channel", use_column_width=True)
    col2.image(I_g, caption="Green Channel", use_column_width=True)
    col3.image(I_b, caption="Blue Channel", use_column_width=True)
    st.image(I_gray, caption="Grayscale", use_column_width=True)
    # Dominant Lowest Channels
    col1, col2 = st.columns(2)
    col1.image(I_dom, caption="Dominant Channel", use_column_width=True)
    col2.image(I_low, caption="Lowest Channel", use_column_width=True)
    # Histogram Enhance
    histVals = []
    bins = list(range(0, 256))
    for b in bins:
        histVals.append(histData[str(b)])
    histFig = plt.figure()
    plt.title("Image Histogram")
    plt.bar(bins, histVals, width=1)
    st.pyplot(histFig)
    st.image(I_histenhanced, caption="Histogram Enhanced", use_column_width=True)

def image_3d_vis():
    # Title
    st.header("Image 3D Vis")

    # Prereq Loaders

    # Load Inputs
    USERINPUT_Image = UI_LoadImage()
    USERINPUT_DepthFuncName = st.selectbox("Depth Detector", list(Image3DVis.DepthLibrary.DEPTH_FUNCS.keys()))

    # Process Inputs
    imgSize = (250, 250)
    keepAspectRatio = True
    I = Image3DVis.ResizeImage(USERINPUT_Image, imgSize, keepAspectRatio)

    DepthFunc = Image3DVis.DepthLibrary.DEPTH_FUNCS[USERINPUT_DepthFuncName]
    options = {}
    options['mods'] = ['Normalise']#, 'Reverse']
    options['NormaliseRange'] = [0, 1]
    options['DepthRange'] = [0, 255]

    DepthScale = 1
    DepthLimits = None
    Depths = Image3DVis.CalculateDepth(I, DepthFunc, options)
    Depths = Depths * DepthScale
    I_depth = np.array(Depths*255, dtype=np.uint8)
    modelFig = Image3DVis.PlotImage3D_Plane(cv2.cvtColor(I, cv2.COLOR_RGB2BGR), Depths, DepthLimits, False, display=False)
    
    # Display Outputs
    # Original
    st.image(USERINPUT_Image, caption="Original", use_column_width=True)
    # Depth Map
    st.image(I_depth, caption="Depth Map", use_column_width=True)
    # 3D Model
    st.pyplot(modelFig)

#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    main()