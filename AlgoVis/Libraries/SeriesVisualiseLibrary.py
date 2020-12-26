'''
Digit based Series generation and visualisation
'''

# Imports
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Libraries import PlotAnimateLibrary as PAL

# Main Functions
# Visualisation Functions
def Series_GroupConvergeVis(ConvergeFunc, computeValues, plotSkip=1, max_iters=-1, titles=['values', 'iters', 'Values vs Iters']):
    iters = []
    for i in tqdm(computeValues):
        trace = ConvergeFunc(i, max_iters=max_iters)
        iters.append(len(trace)-1)
    PAL.List_PlotVisualise(iters[::plotSkip], titles=titles)
    
    return iters

def Series_RangeConvergeVis(ConvergeFunc, computeRange, plotSkip=1, max_iters=-1, titles=['values', 'iters', 'Values vs Iters']):
    iters = []
    for i in tqdm(range(computeRange[0], computeRange[1]+1, computeRange[2])):
        trace = ConvergeFunc(i, max_iters=max_iters)
        iters.append(len(trace)-1)
    PAL.List_PlotVisualise(iters[::plotSkip], titles=titles)
    
    return iters

def Series_ValueConvergeVis(ConvergeFunc, startVal, max_iters=-1, titles=['values', 'iters', 'Values vs Iters']):
    trace = ConvergeFunc(startVal, max_iters=max_iters)
    PAL.List_PlotVisualise(trace, titles=titles)
    return trace

def Series_GroupSubPlotConvergeVis(ConvergeFunc, computeValues, plotSkip=1, max_iters=-1, titles=['values', 'iters', 'Values vs Iters']):
    traces = []
    iters = []
    for i in tqdm(computeValues):
        trace = ConvergeFunc(i, max_iters=max_iters)
        iters.append(len(trace)-1)
        traces.append(trace)

    plt.title(title[2])
    
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

    return iters, traces

def Series_CombinedPlotConvergeVis(ConvergeFunc, computeValues, plotSkip=1, max_iters=-1, titles=['values', 'iters', 'Values vs Iters']):
    traces = []
    iters = []
    for i in tqdm(computeValues):
        trace = ConvergeFunc(i, max_iters=max_iters)
        iters.append(len(trace)-1)
        traces.append(trace)

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
    
    return iters, traces

# Driver Code