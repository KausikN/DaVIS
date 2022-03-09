'''
Library Functions for 3D Plot Visualisation
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
speedUpFactor = 2
rotationSpeed = 3
altDegrees = 30

# Main Functions
# Generation Functions
def GeneratePoints_UniformRandom(N, Limits=[(-15, 15), (-15, 15), (-15, 15)], seed=5):
    np.random.seed(seed)
    x = Limits[0][0] + (Limits[0][1] - Limits[0][0]) * np.random.random(N)
    y = Limits[1][0] + (Limits[1][1] - Limits[1][0]) * np.random.random(N)
    z = Limits[2][0] + (Limits[2][1] - Limits[2][0]) * np.random.random(N)
    pts = np.reshape(np.dstack((x, y, z)), (-1, 3))
    print(pts.shape)
    return pts

def GeneratePoints_Uniform(N, Limits=[(-15, 15), (-15, 15), (-15, 15)]):
    x = np.linspace(Limits[0][0], Limits[0][1], N)
    y = np.linspace(Limits[1][0], Limits[1][1], N)
    z = np.linspace(Limits[2][0], Limits[2][1], N)

    pts = []
    for x0 in x:
        for y0 in y:
            for z0 in z:
                pts.append([x0, y0, z0])
    pts = np.array(pts)

    print(pts.shape)
    return pts

# Visualisation Functions
# initialization function: plot the background of each frame
def InitAnimation():
    global Lines, Pts, x_t, fig, ax
    for line, pt in zip(Lines, Pts):
        line.set_data([], [])
        # line.set_3d_properties(np.array([]))

        pt.set_data([], [])
        # pt.set_3d_properties([])
    return Lines + Pts

# animation function.  This will be called sequentially with the frame number
def UpdateAnimation(i, progressObj=None, frames=1):
    global Lines, Pts, x_t, fig, ax, speedUpFactor, rotationSpeed
    
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (speedUpFactor * i) % x_t.shape[1]

    for line, pt, xi in zip(Lines, Pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(altDegrees, 0.3 * i*rotationSpeed)
    fig.canvas.draw()

    print(i, "done", end='\r')
    if progressObj is not None: progressObj.progress((i+1)/frames)

    return Lines + Pts

def AnimateEffect(EffectFunc, N_trajectories, GeneratorFunc, timeInterval=[0, 4], plotLims=[(-25, 25), (-35, 35), (5, 55)], frames=500, frame_interval=30, plotData=True, saveData={"save": False}, progressObj=None):
    global Lines, Pts, x_t, fig, ax, speedUpFactor
    # Choose random starting points, uniformly distributed from -15 to 15
    startPoints = GeneratorFunc(N_trajectories)
    N_trajectories = startPoints.shape[0]

    # Get Plot Points
    time = np.linspace(timeInterval[0], timeInterval[1], frames*speedUpFactor)
    x_t = np.asarray([EffectFunc(sP, time) for sP in tqdm(startPoints)])

    # Set up figure & 3D axis for animation
    fig = plt.figure(figsize=saveData["figSize"])
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')

    # choose a different color for each trajectory
    colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

    # Set up lines and points
    Lines = sum([ax.plot([], [], [], '-', c=c) for c in colors], [])
    Pts = sum([ax.plot([], [], [], 'o', c=c) for c in colors], [])

    # Prepare the axes limits
    ax.set_xlim(plotLims[0])
    ax.set_ylim(plotLims[1])
    ax.set_zlim(plotLims[2])

    # Set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(altDegrees, 0)

    # Animate
    InitAnim = InitAnimation
    UpdateAnim = functools.partial(UpdateAnimation, progressObj=progressObj, frames=frames)
    anim = animation.FuncAnimation(fig, UpdateAnim, init_func=InitAnim, frames=frames, interval=frame_interval, blit=True)

    # Save as mp4. This requires mplayer or ffmpeg to be installed
    if saveData["save"]:
        if os.path.splitext(saveData["path"])[-1] == '.gif':
            writer = animation.PillowWriter(fps=saveData["fps"])
            anim.save(saveData["path"], writer=writer, )
        else:
            anim.save(saveData["path"], fps=saveData["fps"], extra_args=['-vcodec', 'libx264'])

    if plotData:
        plt.show()


def AnimateEffect_Generic(EffectFunc, Points, Colors, timeInterval=[0, 4], plotLims=[(-25, 25), (-35, 35), (5, 55)], frames=500, frame_interval=30, plotData=True, saveData={"save": False}, progressObj=None):
    global Lines, Pts, x_t, fig, ax, speedUpFactor
    # Choose random starting points, uniformly distributed from -15 to 15
    startPoints = np.array(Points)
    N_trajectories = startPoints.shape[0]

    # Get Plot Points
    time = np.linspace(timeInterval[0], timeInterval[1], frames*speedUpFactor)
    x_t = np.asarray([EffectFunc(sP, time) for sP in tqdm(startPoints)])

    # Set up figure & 3D axis for animation
    fig = plt.figure(figsize=saveData["figSize"])
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')

    # choose a different color for each trajectory
    colors = np.array(Colors)

    # Set up lines and points
    Lines = sum([ax.plot([], [], [], '-', c=c) for c in colors], [])
    Pts = sum([ax.plot([], [], [], 'o', c=c) for c in colors], [])

    # Prepare the axes limits
    ax.set_xlim(plotLims[0])
    ax.set_ylim(plotLims[1])
    ax.set_zlim(plotLims[2])

    # Set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(30, 0)

    # Animate
    InitAnim = InitAnimation
    UpdateAnim = functools.partial(UpdateAnimation, progressObj=progressObj, frames=frames)
    anim = animation.FuncAnimation(fig, UpdateAnim, init_func=InitAnim, frames=frames, interval=frame_interval, blit=True)

    # Save as mp4. This requires mplayer or ffmpeg to be installed
    if saveData["save"]:
        if os.path.splitext(saveData["path"])[-1] == '.gif':
            writer = animation.PillowWriter(fps=saveData["fps"])
            anim.save(saveData["path"], writer=writer, )
        else:
            anim.save(saveData["path"], fps=saveData["fps"], extra_args=['-vcodec', 'libx264'])

    if plotData:
        plt.show()