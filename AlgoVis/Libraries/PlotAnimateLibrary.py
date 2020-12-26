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

# Sample Visualisations
def List_PlotVisualise(values, titles=['', '']):
    # print("No of iters:", len(values)-1)
    values_str = []
    for v in values:
        values_str.append(str(v))
    # print("Trace:", ' '.join(values_str))
    plt.plot(list(range(1, len(values)+1)), values)
    plt.scatter(list(range(1, len(values)+1)), values)
    plt.xlabel(titles[0])
    plt.ylabel(titles[1])
    plt.title(titles[2])
    plt.show()

def ListProgressionPlot_Vis(values):
    frames = len(values)
    ListProgressionPlot_CreatePlotFigure(values)
    return CreatePlotGIF(plotData['fig'], ListProgressionPlot_Update, SimplePlot_Init, frames, True)

def ListProgressionPlot_CreatePlotFigure(values):
    global plotData
    global XData
    global YData

    fig, ax = plt.subplots()
    XData['lim'] = [0, len(values)]
    YData['lim'] = [min(values)-1, max(values)+1]
    XData['data'] = range(len(values))
    YData['data'] = values
    plotData['ax'] = ax
    plotData['fig'] = fig

def SimplePlot_Init():
    global XData
    global YData
    global plotData
    plotData['ax'].set_xlim(XData['lim'][0], XData['lim'][1])
    plotData['ax'].set_ylim(YData['lim'][0], YData['lim'][1])
    plotData['curIndex'] = 0

def ListProgressionPlot_Update(i):
    global XData
    global YData
    global plotData
    
    if plotData['curIndex'] > 0:
        plt.plot([XData['data'][plotData['curIndex']-1], XData['data'][plotData['curIndex']]], [YData['data'][plotData['curIndex']-1], YData['data'][plotData['curIndex']]])
    else:
        plt.plot([XData['data'][plotData['curIndex']]], [YData['data'][plotData['curIndex']]])
    plt.scatter([XData['data'][plotData['curIndex']]], [YData['data'][plotData['curIndex']]])

    plotData['curIndex'] += 1

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