'''
Algorithm Visualisation for Chaos Theory and Lorenz Attractor System
Chaos Theory Link: https://www.youtube.com/watch?v=fDek6cYijxI
Lorenz Attractor Link: https://www.youtube.com/watch?v=VjP90rwpBwU
'''

# Imports
import os
import functools
import numpy as np
from scipy import integrate

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
from tqdm import tqdm

# Main Variables
Lines = []
Pts = []
fig = None
ax = None
x_t = []

# Main Functions
# Algorithm Functions
def lorentz_deriv(pt, t0, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorentz system."""
    x, y, z = pt
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def GenerateRandomPoints_Uniform(N_trajectories, Limits=[-15, 15], seed=1):
    np.random.seed(seed)
    x0 = Limits[0] + (Limits[1] - Limits[0]) * np.random.random((N_trajectories, 3))
    return x0

# Visualisation Functions
# initialization function: plot the background of each frame
def InitLorenzAnimation():
    global Lines, Pts, x_t, fig, ax
    for line, pt in zip(Lines, Pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return Lines + Pts

# animation function.  This will be called sequentially with the frame number
def UpdateLorenzAnimation(i):
    global Lines, Pts, x_t, fig, ax
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]

    for line, pt, xi in zip(Lines, Pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()

    print(i, "done")

    return Lines + Pts

def AnimateLorenzAttractor(N_trajectories, GeneratorFunc, frames=500, frame_interval=30, plotData=True, saveData={"save": False}):
    global Lines, Pts, x_t, fig, ax
    # Choose random starting points, uniformly distributed from -15 to 15
    startPoints = GeneratorFunc(N_trajectories)

    # Solve for the trajectories
    time = np.linspace(0, 4, 1000)
    x_t = np.asarray([integrate.odeint(lorentz_deriv, sP, time) for sP in tqdm(startPoints)])

    # Set up figure & 3D axis for animation
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')

    # choose a different color for each trajectory
    colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

    # Set up lines and points
    Lines = sum([ax.plot([], [], [], '-', c=c) for c in colors], [])
    Pts = sum([ax.plot([], [], [], 'o', c=c) for c in colors], [])

    # Prepare the axes limits
    ax.set_xlim((-25, 25))
    ax.set_ylim((-35, 35))
    ax.set_zlim((5, 55))

    # Set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(30, 0)

    # Animate
    # InitAnim = functools.partial(InitLorenzAnimation, Lines, Pts)
    # UpdateAnim = functools.partial(UpdateLorenzAnimation, Lines=Lines, Pts=Pts, x_t=x_t, ax=ax, fig=fig)
    InitAnim = InitLorenzAnimation
    UpdateAnim = UpdateLorenzAnimation
    anim = animation.FuncAnimation(fig, UpdateAnim, init_func=InitAnim, frames=frames, interval=frame_interval, blit=True)

    # Save as mp4. This requires mplayer or ffmpeg to be installed
    if saveData["save"]:
        if os.path.splitext(saveData["path"])[-1] == '.gif':
            writer = animation.PillowWriter(fps=saveData["fps"])
            anim.save(saveData["path"], writer=writer)
        else:
            anim.save(saveData["path"], fps=saveData["fps"], extra_args=['-vcodec', 'libx264'])

    if plotData:
        plt.show()

# Driver Code
# Params
N_trajectories = 5
GeneratorFunc = GenerateRandomPoints_Uniform

RandomLimits = [-15, 15]
frames = 500
frame_interval = 30

plotData = False
saveData = {"save": True, "path":"AlgoVis/GeneratedVisualisations/LorenzAttractor.gif", "fps": 25}
# Params

# RunCode
AnimateLorenzAttractor(N_trajectories, functools.partial(GeneratorFunc, Limits=RandomLimits), frames=frames, frame_interval=frame_interval, plotData=plotData, saveData=saveData)