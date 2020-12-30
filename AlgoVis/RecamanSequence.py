'''
Algorithm Visualisation for Recaman Sequence
Link: https://www.youtube.com/watch?v=FGC5TdIiT9U
'''

# Imports
import functools
from Libraries import SeriesVisualiseLibrary as SVL

# Main Functions
# Algorithm Functions
def Recaman_Run(startVal, iters=10, minFill=-1):
    values = [startVal]
    curVal = startVal
    curStep = 1
    minAllVisited = 0
    visitedVals = []

    curIter = 0
    while(curIter < iters or (minFill > 0 and minAllVisited < minFill)):
        newVal = curVal
        leftVal = curVal - curStep
        rightVal = curVal + curStep
        if leftVal <= minAllVisited:
            # If Left Path Blocked check Right
            if rightVal in visitedVals:
                # Both Ways Blocked - GO RIGHT BY DEFAULT
                newVal = rightVal
            else:
                # Right available
                newVal = rightVal
        else:
            # If Left Path not blocked, check if not in visited
            if leftVal in visitedVals:
                # Left Path visited, Check Right
                if rightVal in visitedVals:
                    # Both Ways Blocked - GO RIGHT BY DEFAULT
                    newVal = rightVal
                else:
                    # Right available
                    newVal = rightVal
            else:
                # Left Available
                newVal = leftVal

        
        visitedVals.append(newVal)
        # Prune visitedVals and update minAllVisited
        minI = 1
        while(True):
            checkVal = minAllVisited + minI
            if checkVal in visitedVals:
                minI += 1
                minAllVisited += 1
                visitedVals.remove(checkVal)
                # print(minAllVisited)
            else:
                break

        # Update other vars
        curVal = newVal
        values.append(curVal)
        curStep += 1

        curIter += 1

    return values

# Driver Code
# Params
RunFunc = Recaman_Run
# Params

# Run for a value
# Params
startVal = 0
iters = 1000

plotPoints = False
plotLines = True
annotate = False
# Params

# RunCode
ConvergeFuncSingle = functools.partial(RunFunc, iters=iters)
SVL.Series_ValueConvergeVis(ConvergeFuncSingle, startVal, titles=['Iteration', 'Value', " Recaman Sequence for " + str(startVal)], plotLines=plotLines, plotPoints=plotPoints, annotate=annotate)