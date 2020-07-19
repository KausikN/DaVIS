'''
PlotGIFLibrary is a library for generation, editing and viewing of GIFs / Videos of Plot Data
'''

# Imports
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# Main Params
YData = {}
XData = {}
plotData = {}

# Main Functions
def CreatePlotGIF(plotFig, updateFunc, initFunc, frames=np.linspace(0, 2*np.pi, 64), show=False):
    animation = FuncAnimation(plotFig, updateFunc, frames, init_func=initFunc)
    if show:
        plt.show()
    return animation

def SavePlotGIF(animation, savePath, fps=25):
    writer = PillowWriter(fps=fps)
    animation.save(savePath, writer=writer)

# Random Generator Vis - Visualise the distribution of random number generator in python by visualising the frequency distribution
import random

def RandomGenerator_Vis(numRange=(0, 100), frameLim=(0, 100), nframes=100, show=True):
    global XData
    global YData

    XData['lim'] = numRange
    YData['lim'] = (0, 1)
    frames = np.linspace(frameLim[0], frameLim[1], nframes)

    RandomGenerator_CreatePlotFigure()
    YData['maxFreq'] = max(YData['data'])

    return CreatePlotGIF(plotData['fig'], RandomGenerator_PlotUpdate, RandomGenerator_PlotInit, frames, show)

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
numRange = (1, 500)
nframes = 500
frameLim = (0, 1)
show = False
saveGIF = True
savePath = 'GeneratedGIFS/RandomGen_GIF.gif'
saveFPS = 25

# RunCode
animation = RandomGenerator_Vis(numRange, frameLim, nframes, show)
if saveGIF:
    SavePlotGIF(animation, savePath, saveFPS)