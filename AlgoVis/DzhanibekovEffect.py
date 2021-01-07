'''
Algorithm Visualisation for the Dzhanibekov effect
Dzhanibekov effect Link: https://www.youtube.com/watch?v=1VPfZ_XzisU
'''

# Imports
import math
import functools
import numpy as np
from Libraries import Plot3DVisualiseLibrary as P3L

# Main Functions
def Effect_UpwardSpiral(sP, time, ls=50, r=15, rs=1):
    x_t = []
    for t in time:
        z = ls * t
        angle = ((rs * t * 360) % 360)
        rad = (angle / 180) * math.pi
        x = r * math.cos(rad)
        y = r * math.sin(rad)

        x_t.append([sP[0] + x, sP[1] + y, sP[2] + z])
    x_t = np.array(x_t)
    return x_t

# Driver Code
# Params
N_trajectories = 500
GeneratorFunc = P3L.GeneratePoints_UniformRandom

timeInterval = [0, 2.5]
EffectFunc = Effect_UpwardSpiral
saveName = "DzhanibekovEffect"
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
    "figSize": [640, 480]
    }
# Params

# RunCode
saveData["figSize"] = (saveData["figSize"][0]/100, saveData["figSize"][1]/100) # Change FigSize to inches (dpi = 100)
P3L.speedUpFactor = speedUpFactor
P3L.rotationSpeed = rotationSpeed
P3L.AnimateEffect(EffectFunc, N_trajectories, functools.partial(GeneratorFunc, Limits=GenerationLimits), timeInterval=timeInterval, plotLims=plotLims, frames=frames, frame_interval=frame_interval, plotData=plotData, saveData=saveData)