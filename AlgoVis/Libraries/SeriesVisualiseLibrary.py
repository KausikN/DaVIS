'''
Digit based Series generation and visualisation
'''

# Imports
import turtle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Libraries import PlotAnimateLibrary as PAL
from Libraries import TurtleAnimateLibrary as TAL

# Main Functions
# Execution Functions
def Series_GroupExecute(ConvergeFunc, computeValues):
    iters = []
    traces = []
    for i in tqdm(computeValues):
        trace = ConvergeFunc(i)
        traces.append(trace)
        iters.append(len(trace)-1)
    return traces, iters

def Series_RangeExecute(ConvergeFunc, computeRange):
    iters = []
    traces = []
    for i in tqdm(range(computeRange[0], computeRange[1]+1, computeRange[2])):
        trace = ConvergeFunc(i)
        traces.append(trace)
        iters.append(len(trace)-1)
    return traces, iters

def Series_ValueExecute(ConvergeFunc, startVal):
    trace = ConvergeFunc(startVal)
    return trace, len(trace)

# Matplotlib Visualisation Functions
def Series_ValueConvergeVis(ConvergeFunc, startVal, titles=['values', 'iters', 'Values vs Iters'], plotLines=True, plotPoints=True, annotate=False):
    trace, iterCount = Series_ValueExecute(ConvergeFunc, startVal)
    PAL.List_PlotVisualise(trace, titles=titles, plotLines=plotLines, plotPoints=plotPoints, annotate=annotate)

    return trace, iterCount

def Series_GroupConvergeVis(ConvergeFunc, computeValues, plotSkip=1, titles=['values', 'iters', 'Values vs Iters']):
    traces, iters = Series_GroupExecute(ConvergeFunc, computeValues)
    PAL.List_PlotVisualise(iters[::plotSkip], titles=titles)
    
    return traces, iters

def Series_RangeConvergeVis(ConvergeFunc, computeRange, plotSkip=1, titles=['values', 'iters', 'Values vs Iters']):
    traces, iters = Series_RangeExecute(ConvergeFunc, computeRange)
    PAL.List_PlotVisualise(iters[::plotSkip], titles=titles)
    
    return traces, iters

def Series_GroupSubPlotConvergeVis(ConvergeFunc, computeValues, plotSkip=1, titles=['values', 'iters', 'Values vs Iters']):
    traces, iters = Series_GroupExecute(ConvergeFunc, computeValues)

    plt.title(titles[2])
    
    nCols = 5
    nRows = len(computeValues)/nCols
    if nRows > int(nRows):
        nRows = int(nRows) + 1
    nRows = int(nRows)

    for i in range(0, len(traces), plotSkip):
        plt.subplot(nRows, nCols, i+1)
        plt.plot(range(iters[i]+1), traces[i])
        plt.scatter(range(iters[i]+1), traces[i])
        plt.xlabel(titles[0])
        plt.ylabel(titles[1])
    plt.show()

    PAL.List_PlotVisualise(iters[::plotSkip])

    return traces, iters

def Series_CombinedPlotConvergeVis(ConvergeFunc, computeValues, plotSkip=1, titles=['values', 'iters', 'Values vs Iters']):
    traces, iters = Series_GroupExecute(ConvergeFunc, computeValues)

    ax = plt.subplot(1,1,1)
    colors = cm.rainbow(np.linspace(0, 1, len(traces)))
    for i in range(0, len(traces), plotSkip):
        ax.plot(range(iters[i]+1), traces[i], c=colors[i], label=str(computeValues[i]))
        ax.scatter(range(iters[i]+1), traces[i], color=colors[i])
    plt.xlabel(titles[0])
    plt.ylabel(titles[1])
    plt.title(titles[2])

    handles, labels = ax.get_legend_handles_labels()
    # reverse the order
    ax.legend(handles[::-1], labels[::-1])

    plt.show()
    
    return traces, iters

# Turtle Visualisation Functions
def Series_ValueRun_TurtleCurves(trace, titles=['values', 'iters', 'Values vs Iters']):
    # trace, iterCount = Series_ValueExecute(ConvergeFunc, startVal)
    trace = np.array(trace)
    dataRange = np.array([np.min(trace), np.max(trace)])
    trace = ((trace - dataRange[0] - ((dataRange[0] + dataRange[1])/2)) / (dataRange[1] - dataRange[0])) * 700
    
    TAL.List_TurtleValueAlternatingCurves(trace, titles=titles, scale=1)

# Driver Code