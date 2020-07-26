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
'''
# Driver Code
# Params
numRange = (1, 5)
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
'''