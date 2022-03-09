'''
Effect Functions for Animation
'''

# Imports
import math
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Effect Functions
# Effect Functions
def Effect_UpwardSpiral(sP, time, ls=50, r=15, rs=1, **kwargs):
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

def Effect_Translate(sP, time, speed=[-100, 0, 0], **kwargs):
    """
    Params

    timeInterval = [0, 100]
    ImagePointLimits = [(-15, 15), (-27.5, 27.5), (-27.5, 27.5)]
    plotLims = [(-15, 15), (-15, 15), (-15, 15)]
    speedUpFactor = 2

    frames = 125
    """
    x_t = []
    for t in time:
        x = speed[0] * t
        y = speed[1] * t
        z = speed[2] * t

        x_t.append([sP[0] + x, sP[1] + y, sP[2] + z])
    x_t = np.array(x_t)
    return x_t

# Main Vars
EFFECT_FUNCS = {
    "Translate": Effect_Translate,
    "Upward Spiral": Effect_UpwardSpiral
}