'''
Algorithm Visualisation for any Equation
Gravity Link: https://www.youtube.com/watch?v=mezkHBPLZ4A
'''

# Imports
import math
import functools
import numpy as np
from Libraries import Plot3DVisualiseLibrary as P3L

# Main Functions
# Main Equation Functions
def Equation_GenericFunc(sP, time, s=[0, 1, 0], TransformFuncs=None):
    x_t = []
    for t in time:
        x = TransformFuncs[0](t)*s[0]
        y = TransformFuncs[1](t)*s[1]
        z = TransformFuncs[2](t)*s[2]

        x_t.append([sP[0] + x, sP[1] + y, sP[2] + z])
    x_t = np.array(x_t)
    return x_t

# Transform Functions
def TransformFunc_None(x):
    return 0

def TransformFunc_SinCos(x, coeff=[1, 1]):
    return (math.sin(x)*coeff[0] + math.cos(x)*coeff[1])

def TransformFunc_Linear(x, coeff=1):
    return coeff*x

# Driver Code
# Params
N_trajectories = 1
GeneratorFunc = P3L.GeneratePoints_Uniform

timeInterval = [0, 50]
Transforms = [
    TransformFunc_None,
    functools.partial(TransformFunc_Linear, coeff=0.5),
    functools.partial(TransformFunc_SinCos, coeff=[10, 0])
    ]
EffectFunc = functools.partial(Equation_GenericFunc, s=[0, 1, 1], TransformFuncs=Transforms)
saveName = "EquationVis"
GenerationLimits = [(0, 0), (-17, 17), (0, 0)]
plotLims = [(-15, 15), (-15, 15), (-15, 15)]
speedUpFactor = 2

frames = 250
frame_interval = 30
rotationSpeed = 0

plotData = False
saveData = {
    "save": True,
    "path":"AlgoVis/GeneratedVisualisations/" + saveName + "_" +
        ("Uniform" if GeneratorFunc == P3L.GeneratePoints_Uniform else "Random") + ".gif",
    "fps": 30,
    "figSize": [320, 240]
    }
# Params

# RunCode
saveData["figSize"] = (saveData["figSize"][0]/100, saveData["figSize"][1]/100) # Change FigSize to inches (dpi = 100)
P3L.speedUpFactor = speedUpFactor
P3L.rotationSpeed = rotationSpeed
P3L.AnimateEffect(EffectFunc, N_trajectories, functools.partial(GeneratorFunc, Limits=GenerationLimits), timeInterval=timeInterval, plotLims=plotLims, frames=frames, frame_interval=frame_interval, plotData=plotData, saveData=saveData)