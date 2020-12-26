'''
Digit based Series generation and visualisation
'''

# Imports
import numpy as np
from tqdm import tqdm
from Libraries import PlotAnimateLibrary as PAL
from Libraries import SeriesVisualiseLibrary as SVL

# Main Functions
# Converge Functions
def DigitSumSeries_Converge(startVal, max_iters=-1):
    values = [np.abs(startVal)]
    sign = np.sign(startVal)
    curVal = np.abs(startVal)
    n_iters = 0
    while(len(str(curVal)) > 1):
        curVal = np.sum(np.array(list(str(curVal))).astype(int))

        values.append(curVal)

        n_iters += 1
        if n_iters > max_iters and max_iters > -1:
            break

    values = list(np.array(values) * sign)

    return values

def DigitMultiplySeries_Converge(startVal, max_iters=-1):
    values = [np.abs(startVal)]
    sign = np.sign(startVal)
    curVal = np.abs(startVal)
    n_iters = 0
    while(len(str(curVal)) > 1):
        newCurVal = 1
        for v in np.array(list(str(curVal))).astype(int):
            newCurVal *= v
        curVal = newCurVal

        values.append(curVal)

        n_iters += 1
        if n_iters > max_iters and max_iters > -1:
            break

    values = list(np.array(values) * sign)

    return values

# Driver Code
# Params
ConvergeFunc = DigitMultiplySeries_Converge
# Params

# Converge for a value
# Params
startVal = 1123314
max_iters = -1
# Params

# RunCode
SVL.Series_ValueConvergeVis(ConvergeFunc, startVal, max_iters, titles=['Iteration', 'Value', " Digit Multiply Convergence for " + str(startVal)])

# Converge Over Many Values
# Params
computeRange = [10, 100, 1]
plotSkip = 1
# Params

# RunCode
iters = SVL.Series_RangeConvergeVis(ConvergeFunc, computeRange, plotSkip=plotSkip, titles=['Start Value', 'Convergence Iterations Count', 'Values vs Digit Multiply Convergence Time'])