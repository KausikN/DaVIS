"""
Stream lit GUI for hosting DaVIS
"""

# Imports
import os
import cv2
import functools
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import json

from ImageVis import Image2DVis
from ImageVis import Image3DVis
from ImageVis import ImagePointEffect
from AudioVis import Audio2DVis

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
PATHS = {
    "cache": "StreamLitGUI/CacheData/Cache.json",
    "default": {
        "example": {
            "image": "TestData/TestImgs/Arch.jpeg",
            "audio": "TestData/TestAudio/DN.mp3",
            "video": "TestData/TestVideos/Test_Animation.wmv"
        },
        "save": {
            "pointgif": "StreamLitGUI/CacheData/PointGif.gif",
            "audio": "StreamLitGUI/CacheData/CacheAudio.mp3",
            "cutaudio": "StreamLitGUI/CacheData/CacheCutAudio.wav"
        },
        "url": {
            "video": "http://192.168.0.102:8080/shot.jpg"
        }
    }
}

# Util Vars
CACHE = {}
INPUTREADERS_VIDEO = Image3DVis.INPUTREADERS_VIDEO

# Util Functions
def LoadCache():
    global CACHE
    CACHE = json.load(open(PATHS["cache"], 'r'))

def SaveCache():
    global CACHE
    json.dump(CACHE, open(PATHS["cache"], 'w'), indent=4)

# Main Functions
@st.cache
def ImageVis_2D(USERINPUT_Image):
    # RGB Gray
    I_r, I_g, I_b = Image2DVis.ImageVis_RGBChannels(USERINPUT_Image, display=False)
    I_gray = Image2DVis.ImageVis_Greyscale(USERINPUT_Image, display=False)
    # Dominant Lowest Channels
    I_dom = Image2DVis.ImageVis_DominantChannel(USERINPUT_Image, display=False)
    I_low = Image2DVis.ImageVis_LowestChannel(USERINPUT_Image, display=False)
    # Histogram Enhance
    histData = Image2DVis.ImageHistogram_2D(USERINPUT_Image, bins=list(range(0, 256)), display=False)
    I_histenhanced = Image2DVis.ImageHistogram_Enhance(USERINPUT_Image, histData, selectRange=[0, 255], display=False)
    return I_r, I_g, I_b, I_gray, I_dom, I_low, histData, I_histenhanced

@st.cache
def ImageVis_3D(USERINPUT_Image, imgSize, keepAspectRatio, DepthFunc, DepthOptions, DepthScale, DepthLimits):
    I = Image3DVis.ResizeImage(USERINPUT_Image, imgSize, keepAspectRatio)
    Depths = Image3DVis.CalculateDepth(I, DepthFunc, DepthOptions)
    Depths = Depths * DepthScale
    I_depth = np.array(Depths*255, dtype=np.uint8)
    modelFig = Image3DVis.PlotImage3D_Plane(cv2.cvtColor(I, cv2.COLOR_RGB2BGR), Depths, DepthLimits, display=False)
    return I, Depths, I_depth, modelFig

def AudioVis_2D(AUDIO, SAMPLE_RATE):
    fig_WAVE = Audio2DVis.DisplayAudio_WavePlot(AUDIO, SAMPLE_RATE, display=False)
    FREQUENCIES, TIMES, SPECTROGRAM = Audio2DVis.GetFrequencyData(AUDIO, SAMPLE_RATE)
    spectrogram_min, spectrogram_max = SPECTROGRAM.min(), SPECTROGRAM.max()
    SPECTROGRAM_norm = (SPECTROGRAM - spectrogram_min) / (spectrogram_max - spectrogram_min)
    SPECTROGRAM_norm = np.array(SPECTROGRAM_norm * 255, dtype=np.uint8)
    fig_MAXFREQ = Audio2DVis.DisplayMaxFrequencyGraph(FREQUENCIES, TIMES, SPECTROGRAM, plotSkip=1, display=False)
    return fig_WAVE, fig_MAXFREQ, SPECTROGRAM_norm

def ImagePoint_Effect(USERINPUT_Image, USERINPUT_EffectName, timeInterval, frames):
    imgSize = (30, 30)
    keepAspectRatio = False
    I = Image3DVis.ResizeImage(USERINPUT_Image, imgSize, keepAspectRatio)
    DepthOptions = {
        "mods": ['Normalise'],#, 'Reverse']
        "NormaliseRange": [0, 1],
        "DepthRange": [0, 255]
    }
    DepthFunc = functools.partial(ImagePointEffect.DepthFunc_GrayScaleDepth, options=DepthOptions)

    EffectOptions = {
        # Upward Spiral
        "ls": 50,
        "r": 15,
        "rs": 1,
        # Translate
        "speed": [-100, 0, 0]
    }
    EffectFunc = functools.partial(ImagePointEffect.EffectFunctions.EFFECT_FUNCS[USERINPUT_EffectName], **EffectOptions)
    
    ImagePointLimits = [(-15, 15), (-15, 15), (-15, 15)]
    plotLims = [(-30, 30), (-30, 30), (0, 55)]
    speedUpFactor = 1
    frame_interval = 30
    rotationSpeed = 0
    altDegrees = 30

    plotData = False
    saveData = {
        "save": True,
        "path": PATHS["default"]["save"]["pointgif"],
        "fps": 30,
        "figSize": [320, 240]
    }
    saveData["figSize"] = (saveData["figSize"][0]/100, saveData["figSize"][1]/100) # Change FigSize to inches (dpi = 100)
    ImagePointEffect.P3L.speedUpFactor = speedUpFactor
    ImagePointEffect.P3L.rotationSpeed = rotationSpeed
    ImagePointEffect.P3L.altDegrees = altDegrees

    Points, Colors = ImagePointEffect.Image2PointsColors(I, DepthFunc, ImagePointLimits)
    ImagePointEffect.P3L.AnimateEffect_Generic(
        EffectFunc, Points, Colors, timeInterval=timeInterval, plotLims=plotLims, frames=frames, frame_interval=frame_interval, 
        plotData=plotData, saveData=saveData,
        progressObj=st.progress(0.0)
    )

def VideoVis_3D(USERINPUT_Image, imgSize, keepAspectRatio, DepthFunc, DepthOptions, DepthScale, DepthLimits, 
                Widgets, applyTexture=False, fig=None):
    I = Image3DVis.ResizeImage(USERINPUT_Image, imgSize, keepAspectRatio)
    Depths = Image3DVis.CalculateDepth(I, DepthFunc, DepthOptions)
    Depths = Depths * DepthScale
    if applyTexture:
        texture = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
    else:
        texture = np.ones((I.shape[0], I.shape[1], 3), dtype=np.uint8) * 128
    modelFig = Image3DVis.PlotImage3D_Plane(texture, Depths, DepthLimits, fig=fig, display=False)
    # Display Outputs
    # Original
    Widgets["input"].image(USERINPUT_Image, caption="Original", use_column_width=True)
    # 3D Model
    if fig is None:
        Widgets["output"] = Widgets["output"].plotly_chart(modelFig, use_container_width=True)
    else:
        Widgets["output"].plotly_chart(modelFig, use_container_width=True)
    

    return modelFig

# UI Functions
def UI_LoadImage():
    USERINPUT_ImageData = st.file_uploader("Upload Image", ['png', 'jpg', 'jpeg', 'bmp'])
    if USERINPUT_ImageData is not None:
        USERINPUT_ImageData = USERINPUT_ImageData.read()
    else:
        USERINPUT_ImageData = open(PATHS["default"]["example"]["image"], 'rb').read()
    USERINPUT_ImageData = cv2.imdecode(np.frombuffer(USERINPUT_ImageData, np.uint8), cv2.IMREAD_COLOR)
    USERINPUT_Image = cv2.cvtColor(USERINPUT_ImageData, cv2.COLOR_BGR2RGB)
    return USERINPUT_Image

def UI_LoadAudio():
    USERINPUT_AudioData = st.file_uploader("Upload Audio", ['mp3'])
    if USERINPUT_AudioData is not None:
        open(PATHS["default"]["save"]["audio"], 'wb').write(USERINPUT_AudioData.read())
        USERINPUT_AudioPath = PATHS["default"]["save"]["audio"]
    else:
        USERINPUT_AudioPath = PATHS["default"]["example"]["audio"]

    col1, col2 = st.columns(2)
    OFFSET = col1.number_input("Start Time", 0.0, 100.0, 0.0, 0.1)
    DURATION = col2.number_input("Duration", 0.1, 10.0, 1.0, 0.1)
    AUDIO, SAMPLE_RATE = Audio2DVis.LoadAudio(USERINPUT_AudioPath, duration=DURATION, offset=OFFSET)
    Audio2DVis.SaveAudio(AUDIO, SAMPLE_RATE, PATHS["default"]["save"]["cutaudio"])

    return AUDIO, SAMPLE_RATE

def UI_LoadVideo():
    USERINPUT_VideoInputChoice = st.selectbox("Select Video Input Source", list(INPUTREADERS_VIDEO.keys()))
    USERINPUT_VideoReader = INPUTREADERS_VIDEO[USERINPUT_VideoInputChoice]

    # Upload Video File
    if USERINPUT_VideoInputChoice == "Upload Video File":
        USERINPUT_VideoPath = st.file_uploader("Upload Video", ['avi', 'mp4', 'wmv'])
        if USERINPUT_VideoPath is None:
            USERINPUT_VideoPath = PATHS["default"]["example"]["video"]
        USERINPUT_VideoReader = functools.partial(USERINPUT_VideoReader, USERINPUT_VideoPath)
    # Video URL
    elif USERINPUT_VideoInputChoice == "Video URL":
        USERINPUT_VideoURL = st.text_input("Video URL", PATHS["default"]["url"]["video"])
        USERINPUT_VideoReader = functools.partial(USERINPUT_VideoReader, USERINPUT_VideoURL)
    # Webcam
    else:
        pass

    USERINPUT_Video = USERINPUT_VideoReader()
    
    return USERINPUT_Video

# Repo Based Functions
def image_2d_vis():
    # Title
    st.header("Image 2D Vis")

    # Prereq Loaders

    # Load Inputs
    USERINPUT_Image = UI_LoadImage()
    # Original
    st.markdown("## Image")
    st.image(USERINPUT_Image, caption="Original", use_column_width=True)

    # Process Inputs
    if st.button("Visualise"):
        I_r, I_g, I_b, I_gray, I_dom, I_low, histData, I_histenhanced = ImageVis_2D(USERINPUT_Image)

        # Display Outputs
        st.markdown("## Visualisations")
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
        st.plotly_chart(histFig, use_container_width=True)
        st.image(I_histenhanced, caption="Histogram Enhanced", use_column_width=True)

def image_3d_vis():
    # Title
    st.header("Image 3D Vis")

    # Prereq Loaders

    # Load Inputs
    USERINPUT_Image = UI_LoadImage()
    USERINPUT_DepthFuncName = st.selectbox("Depth Detector", list(Image3DVis.DepthLibrary.DEPTH_FUNCS.keys()))
    USERINPUT_DepthScale = st.number_input("Depth Scale", 0.0, 2.0, 0.5, 0.1)
    # Original
    st.markdown("## Image")
    st.image(USERINPUT_Image, caption="Original", use_column_width=True)

    # Process Inputs
    if st.button("Visualise"):
        imgSize = (250, 250)
        keepAspectRatio = True
        
        DepthFunc = Image3DVis.DepthLibrary.DEPTH_FUNCS[USERINPUT_DepthFuncName]
        DepthOptions = {
            'mods': ['Normalise'],#, 'Reverse']
            'NormaliseRange': [0, 1],
            'DepthRange': [0, 255]
        }
        DepthScale = USERINPUT_DepthScale
        DepthLimits = None

        I, Depths, I_depth, modelFig = ImageVis_3D(USERINPUT_Image, imgSize, keepAspectRatio, DepthFunc, DepthOptions, DepthScale, DepthLimits)
        
        # Display Outputs
        st.markdown("## Visualisations")
        # Depth Map
        st.image(I_depth, caption="Depth Map", use_column_width=True)
        # 3D Model
        st.plotly_chart(modelFig, use_container_width=True)

def audio_2d_vis():
    # Title
    st.header("Audio 2D Vis")

    # Prereq Loaders

    # Load Inputs
    AUDIO, SAMPLE_RATE = UI_LoadAudio()
    # Original
    st.markdown("## Audio")
    st.audio(open(PATHS["default"]["save"]["cutaudio"], "rb").read(), format="audio/wav", start_time=0)

    # Process Inputs
    if st.button("Visualise"):
        fig_WAVE, fig_MAXFREQ, SPECTROGRAM_norm = AudioVis_2D(AUDIO, SAMPLE_RATE)

        # Display Outputs
        st.markdown("## Visualisations")
        st.plotly_chart(fig_WAVE, use_container_width=True)
        st.image(SPECTROGRAM_norm, caption="Spectrogram", use_column_width=True)
        st.plotly_chart(fig_MAXFREQ, use_container_width=True)

def image_point_vis():
    # Title
    st.header("Image Point Vis")

    # Prereq Loaders

    # Load Inputs
    USERINPUT_Image = UI_LoadImage()
    USERINPUT_EffectName = st.selectbox("Effect", list(ImagePointEffect.EffectFunctions.EFFECT_FUNCS.keys()))
    col1, col2, col3 = st.columns(3)
    USERINPUT_TimeIntervalStart = col1.number_input("Time Interval Start", 0.0, 100.0, 0.0, 0.1)
    USERINPUT_TimeIntervalEnd = col2.number_input("Time Interval End", 0.1, 100.0, 1.0, 0.1)
    USERINPUT_frames = col3.number_input("Num Frames", 1, 90, 30, 1)
    # Original
    st.markdown("## Image")
    st.image(USERINPUT_Image, caption="Original", use_column_width=True)

    # Process Inputs
    if st.button("Visualise"):
        timeInterval = [USERINPUT_TimeIntervalStart, USERINPUT_TimeIntervalEnd]
        ImagePoint_Effect(USERINPUT_Image, USERINPUT_EffectName, timeInterval, USERINPUT_frames)

        # Display Outputs
        st.markdown("## Visualisations")
        # Point Effect
        st.image(PATHS["default"]["save"]["pointgif"], caption="Point Effect", use_column_width=True)

def video_3d_vis():
    # Title
    st.header("Video 3D Vis")

    # Prereq Loaders

    # Load Inputs
    USERINPUT_Video = UI_LoadVideo()
    USERINPUT_DepthFuncName = st.selectbox("Depth Detector", list(Image3DVis.DepthLibrary.DEPTH_FUNCS.keys()))
    USERINPUT_DepthScale = st.number_input("Depth Scale", 0.0, 2.0, 0.5, 0.1)
    USERINPUT_applyTexture = st.checkbox("Apply Texture", value=True)

    # Process Inputs
    if st.button("Visualise"):
        imgSize = (250, 250)
        keepAspectRatio = True
        
        DepthFunc = Image3DVis.DepthLibrary.DEPTH_FUNCS[USERINPUT_DepthFuncName]
        DepthOptions = {
            'mods': ['Normalise'],#, 'Reverse']
            'NormaliseRange': [0, 1],
            'DepthRange': [0, 255]
        }
        DepthScale = USERINPUT_DepthScale
        DepthLimits = None

        Widgets = {
            "input": st.empty(),
            "output": st.empty()
        }
        VisFunc = functools.partial(
            VideoVis_3D, imgSize=imgSize, keepAspectRatio=keepAspectRatio, 
            DepthFunc=DepthFunc, DepthOptions=DepthOptions, DepthScale=DepthScale, DepthLimits=DepthLimits, 
            Widgets=Widgets, applyTexture=USERINPUT_applyTexture
        )

        Image3DVis.VideoVis_Framewise(VisFunc, USERINPUT_Video, None, -1)
    

#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    main()