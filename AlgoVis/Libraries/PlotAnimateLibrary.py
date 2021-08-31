'''
PlotGIFLibrary is a library for generation, editing and viewing of GIFs / Videos of Plot Data
'''

# Imports
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

import os
import cv2
from PIL import Image
from tqdm import tqdm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Func Animation Plot Visualisation ######################################################################################
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

# List Visualisations
def List_PlotVisualise(values, titles=['', '', ''], plotLines=True, plotPoints=True, annotate=False):
    fig, ax = plt.subplots()
    if plotLines:
        ax.plot(list(range(1, len(values)+1)), values)
    if plotPoints:
        ax.scatter(list(range(1, len(values)+1)), values)
    plt.xlabel(titles[0])
    plt.ylabel(titles[1])
    plt.title(titles[2])
    # print("No of iters:", len(values)-1)
    values_str = []
    for i in range(len(values)):
        values_str.append(str(values[i]))
        if annotate:
            ax.annotate(str(values[i]), (i+1, values[i]))
    # print("Trace:", ' '.join(values_str))
    
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
# Func Animation Plot Visualisation ######################################################################################

# Image by Image Plot Visualisation ######################################################################################
# Main Vars
figsize = (6.4, 4.8)
dpi = 100.0
fig = Figure(figsize=figsize, dpi=dpi)
canvas = FigureCanvasAgg(fig)

# Main Functions
def SaveImages2GIF(frames, savePath, fps=20.0, size=(640, 480)):
    frames_updated = []
    for frame in tqdm(frames):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_updated.append(Image.fromarray(frame))
        
    if os.path.splitext(savePath)[-1] == '.gif':
        extraFrames = []
        if len(frames_updated) > 1:
            extraFrames = frames_updated[1:]
        frames_updated[0].save(savePath, save_all=True, append_images=extraFrames, format='GIF', loop=0)
    else:
        out = cv2.VideoWriter(savePath, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
        for frame in frames_updated:
            out.write(frame)
        out.release()

# Bar Visualisations
def ListUpdatePlot_Bar(listTrace, showAxis=True):
    global fig
    global canvas

    PlotIs = []
    for i, listVals in enumerate(listTrace):
        fig.clear(True)
        ax = fig.add_subplot(111)
        if not showAxis:
            ax.axis('off')
            ax.margins(0)
            fig.tight_layout(pad=0)

        ax.bar(range(len(listVals)), listVals[:])
        ax.title.set_text(str(i))

        canvas.draw()
        buf = canvas.buffer_rgba()
        I_effect = cv2.cvtColor(np.asarray(buf), cv2.COLOR_RGBA2RGB)
        PlotIs.append(I_effect)

    return PlotIs

# Image by Image Plot Visualisation ######################################################################################


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