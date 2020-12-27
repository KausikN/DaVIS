'''
Algorithm Visualisation for Fibonacci and related series
Link: 
'''

# Imports
import functools
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from Libraries import PlotAnimateLibrary as PAL
from Libraries import SeriesVisualiseLibrary as SVL

# Main Functions
# Algorithm Functions
def Fibonacci_Standard(iters, startVal_1=1, startVal_2=1):
    values = [startVal_1, startVal_2]
    curVals = [startVal_1, startVal_2]
    curIter = 0
    while(curIter < iters):
        curVals = [curVals[1], curVals[0] + curVals[1]]
        values.append(curVals[1])
        curIter += 1

    return values

def Fibonacci_GenericLength(iters, startVals=[1, 1]):
    values = list(startVals)
    curVals = list(startVals)
    curIter = 0
    while(curIter < iters):
        curVals = curVals[1:] + [(np.sum(np.array(curVals)))]
        values.append(curVals[-1])
        curIter += 1

    return values

def Fibonacci_GenericFunc(iters, NextFunc, startVals=[1, 1]):
    values = list(startVals)
    curVals = list(startVals)
    curIter = 0
    while(curIter < iters or iters == -1):
        curVals, newVal, stop = NextFunc(curVals, curIter, iters)
        values.append(newVal)
        curIter += 1
        if stop:
            break

    return values

# Generic Fibonacci Functions
def FibonacciGenericFunc_OddAdd(curVals, curIter, iters):
    curVals = curVals[1:] + [(np.sum(np.array(curVals)[1::2]))]
    newVal = curVals[-1]
    return curVals, newVal, False

def FibonacciGenericFunc_EvenAdd(curVals, curIter, iters):
    curVals = curVals[1:] + [(np.sum(np.array(curVals)[::2]))]
    newVal = curVals[-1]
    return curVals, newVal, False

def FibonacciGenericFunc_InverseAdd(curVals, curIter, iters):
    curVals = curVals[1:] + [(np.sum(1/np.array(curVals)))]
    newVal = curVals[-1]
    return curVals, newVal, False

def FibonacciGenericFunc_ReverseGenericLength(curVals, curIter, iters, positiveOnly=False):
    curVals = curVals[1:] + [curVals[0] - (np.sum(np.array(curVals[1:])))]
    newVal = curVals[-1]
    return curVals, newVal, (newVal <= 0) and (positiveOnly)

# Visualisation Functions


# Driver Code
# Standard Fibonacci
# Params
FibonacciFunc = Fibonacci_Standard
startVal_1 = 1
startVal_2 = 1
iters = 50
# Params

# RunCode
FibonacciFuncStandard = functools.partial(FibonacciFunc, startVal_1=startVal_1, startVal_2=startVal_2)
SVL.Series_ValueConvergeVis(FibonacciFuncStandard, iters, titles=['Iteration', 'Value', " Standard Fibonacci Series for " + str(startVal_1) + "," + str(startVal_2)], annotate=True)

# Generic Length Fibonacci
# Params
FibonacciFunc = Fibonacci_GenericLength
startVals = [1, 2, 3]
iters = 50
# Params

# RunCode
FibonacciFuncGenericLength = functools.partial(FibonacciFunc, startVals=startVals)
SVL.Series_ValueConvergeVis(FibonacciFuncGenericLength, iters, titles=['Iteration', 'Value', " Generic Length Fibonacci Series for " + ','.join(np.array(startVals).astype(str))], annotate=True)

# Generic Length Fibonacci
# Params
FibonacciFunc = Fibonacci_GenericFunc
GenericFunc = functools.partial(FibonacciGenericFunc_ReverseGenericLength, positiveOnly=True)
startVals = [13, 8]
iters = -1
# Params

# RunCode
FibonacciFuncGenericFunc = functools.partial(FibonacciFunc, NextFunc=GenericFunc, startVals=startVals)
SVL.Series_ValueConvergeVis(FibonacciFuncGenericFunc, iters, titles=['Iteration', 'Value', " Generic Function Fibonacci Series for " + ','.join(np.array(startVals).astype(str))], annotate=True)