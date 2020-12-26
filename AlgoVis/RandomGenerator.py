'''
PlotGIFLibrary is a library for generation, editing and viewing of GIFs / Videos of Plot Data
'''

# Imports
import random
import matplotlib.pyplot as plt
import numpy as np

from Libraries import PlotAnimateLibrary as PAL

# Main Params
YData = {}
XData = {}
plotData = {}

# Main Functions
# Random Generator Vis - Visualise the distribution of random number generator in python by visualising the frequency distribution


def RandomGenerator_Vis(numRange=(0, 100), frameLim=(0, 100), nframes=100, show=True):
    global XData
    global YData

    XData['lim'] = numRange
    YData['lim'] = (0, 1)
    frames = np.linspace(frameLim[0], frameLim[1], nframes)

    RandomGenerator_CreatePlotFigure()
    YData['maxFreq'] = max(YData['data'])

    return PAL.CreatePlotGIF(plotData['fig'], RandomGenerator_PlotUpdate, RandomGenerator_PlotInit, frames, show)

def RandomGenerator_CreatePlotFigure():
    global plotData
    global XData
    global YData

    fig, ax = plt.subplots()
    XData['data'] = range(XData['lim'][0], XData['lim'][1] + 1)
    YData['data'] = [0]*(XData['lim'][1] - XData['lim'][0] + 1)
    rects = plt.bar(XData['data'], YData['data'])
    plotData['plotVar'] = rects
    plotData['ax'] = ax
    plotData['fig'] = fig

def RandomGenerator_PlotInit():
    global XData
    global YData
    global plotData
    plotData['ax'].set_xlim(XData['lim'][0], XData['lim'][1])
    plotData['ax'].set_ylim(YData['lim'][0], YData['lim'][1])

def RandomGenerator_PlotUpdate(i):
    print(i)
    global XData
    global YData
    global plotData
    newVal = random.randint(XData['lim'][0], XData['lim'][1])
    newVal_Index = XData['data'].index(newVal)
    YData['data'][newVal_Index] += 1
    if YData['data'][newVal_Index] > YData['maxFreq']:
        YData['maxFreq'] = YData['data'][newVal_Index]
        YData['lim'] = (YData['lim'][0], YData['maxFreq'] + 1)
        plotData['ax'].set_ylim(YData['lim'][0], YData['lim'][1])
    plotData['plotVar'][newVal_Index].set_height(YData['data'][newVal_Index])


# Driver Code
# Params
numRange = (1, 5)
nframes = 500
frameLim = (0, 1)
show = True
saveGIF = False
savePath = 'GeneratedGIFS/RandomGen_GIF.gif'
saveFPS = 25

# RunCode
animation = RandomGenerator_Vis(numRange, frameLim, nframes, show)
if saveGIF:
    PAL.SavePlotGIF(animation, savePath, saveFPS)