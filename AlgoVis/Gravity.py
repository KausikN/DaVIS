'''
Algorithm Visualisation for Gravity
Gravity Link: https://www.youtube.com/watch?v=mezkHBPLZ4A
'''

# Imports
import math
import functools
import numpy as np
from Libraries import Plot3DVisualiseLibrary as P3L

# Main Functions
def Effect_Gravity(sP, time, vi=[0, 0, 0], g=9.8):
    x_t = []
    for t in time:
        z = - ((0.5)*g*(t**2) + vi[1]*t)
        x = vi[0]*t
        y = vi[2]*t

        x_t.append([sP[0] + x, sP[1] + y, sP[2] + z])
    x_t = np.array(x_t)
    return x_t

# Driver Code
# Params
initial_velocity = [0, 0, 10]
g = 9.8

N_trajectories = 1
GeneratorFunc = P3L.GeneratePoints_UniformRandom

timeInterval = [0, 10]
EffectFunc = functools.partial(Effect_Gravity, vi=initial_velocity, g=g)
saveName = "Gravity"
GenerationLimits = [(-15, 15), (-15, 15), (-15, 15)]
plotLims = [(-15, 15), (-15, 15), (-1, 75)]
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