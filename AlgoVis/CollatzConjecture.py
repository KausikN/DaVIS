'''
Algorithm Visualisation for rthe Collatz Conjecture
Link: https://www.youtube.com/watch?v=5mFpVDpKX70
'''

# Imports
import functools
import matplotlib.pyplot as plt
from tqdm import tqdm
from Libraries import PlotAnimateLibrary as PAL
from Libraries import SeriesVisualiseLibrary as SVL

# Main Functions
# Algorithm Functions
def CollatzConjecture_Converge(startVal, max_iters=-1):
    values = [startVal]
    curVal = startVal
    n_iters = 0
    while(curVal > 1):
        if curVal % 2 == 0:
            curVal = int(round(curVal / 2))
        else:
            curVal = 3*curVal + 1

        values.append(curVal)

        n_iters += 1
        if n_iters > max_iters and max_iters > -1:
            break

    return values

# Driver Code
# Params
ConvergeFunc = CollatzConjecture_Converge
# Params

# Converge for a value
# Params
startVal = 5798
max_iters = -1
# Params

# RunCode
SVL.Series_ValueConvergeVis(ConvergeFunc, startVal, max_iters, titles=['Iteration', 'Value', " Collatz Convergence for " + str(startVal)])

# Converge Over Many Values
# Params
computeRange = [10, 100, 1]
plotSkip = 1
# Params

# RunCode
iters = SVL.Series_RangeConvergeVis(ConvergeFunc, computeRange, plotSkip=plotSkip, titles=['Start Value', 'Convergence Iterations Count', 'Values vs Collatz Convergence Time'])