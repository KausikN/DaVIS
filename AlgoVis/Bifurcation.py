'''
Algorithm Visualisation for Bifurcation Graph
Video: https://www.youtube.com/watch?v=ovJcsL7vyrk
'''

# Imports
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Libraries import SeriesVisualiseLibrary as SVL

# Main Functions
# Algorithm Functions
def Converge_Logistic(r, startVal=0.5, c=0, max_iters=-1, equilibriumRoundoff=4, valueLimits=[-10000000.0, 10000000.0]):
    values = [startVal]
    curVal = startVal
    n_iters = 0
    while(True):
        # print(r, curVal, r*(curVal)*(1-curVal) + c)
        curVal = r*(curVal)*(1-curVal) + c

        n_iters += 1
        # Exceeding Max Iters
        if n_iters > max_iters and max_iters > -1:
            break
        
        if curVal < valueLimits[0]:
            print(r, curVal, "Low")
            # curVal = valueLimits[0]
        elif curVal > valueLimits[1]:
            print(r, curVal, "High")
            # curVal = valueLimits[1]

        # No change - reached equilibrium
        if round(values[-1], equilibriumRoundoff) == round(curVal, equilibriumRoundoff):
            values.append(curVal)
            break
        values.append(curVal)

    return values

# Visualisation Functions
def FindEquilibriumPopulations(trace, equilibriumRoundoff):
    eqPops = [trace[-1]]
    finalPopRoundedOff = round(trace[-1], equilibriumRoundoff)
    for pop in trace[:-1][::-1]:
        if round(pop, equilibriumRoundoff) == finalPopRoundedOff:
            break
        eqPops.append(pop)

    return eqPops

def BifurcatePlot(rValues, eqPops, titles=['', '', '']):
    # plt.plot(rValues, eqPops)
    colors = cm.rainbow(np.linspace(0, 1, len(rValues)))
    for i in range(len(rValues)):
        plt.scatter([rValues[i]]*len(eqPops[i]), eqPops[i], color=colors[i])
    plt.xlabel(titles[0])
    plt.ylabel(titles[1])
    plt.title(titles[2])
    plt.show()

# Driver Code
# Params
ConvergeFunc = Converge_Logistic

startVal = 0.5
r = 2.6
c = 0.5

max_iters = 500
equilibriumRoundoff = 4
valueLimits = [-float(pow(10, 100)), float(pow(10, 100))]
# Params

# Converge Over Many Values
# Params
r_computeRange = (np.array(range(0, 50 + 1))/10)
# computeRange = np.array([3.0])
plotSkip = 1
# Params

# RunCode
ConvergeFuncManyValues = functools.partial(ConvergeFunc, startVal=startVal, c=c, max_iters=max_iters, equilibriumRoundoff=equilibriumRoundoff, valueLimits=valueLimits)
traces, iters = SVL.Series_CombinedPlotConvergeVis(ConvergeFuncManyValues, r_computeRange, plotSkip=plotSkip, titles=['Iteration', 'Value', 'Population Convergence'])

# Bifurcate Plot
eqPopulations = []
for trace in traces:
    eqPopulations.append(FindEquilibriumPopulations(trace, equilibriumRoundoff=equilibriumRoundoff))
BifurcatePlot(r_computeRange, np.array(eqPopulations), titles=['r', 'Equilibrium Population', 'Bifurcation Plot'])