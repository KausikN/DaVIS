'''
Algorithm Visualisation for Chaos Theory and Lorenz Attractor System
Chaos Theory Link: https://www.youtube.com/watch?v=fDek6cYijxI
Lorenz Attractor Link: https://www.youtube.com/watch?v=VjP90rwpBwU
Aizawa Attractor Link: https://www.youtube.com/watch?v=RBqbQUu-p00
Newton-Leipnip Attractor Link: 

Other Attractors Link: https://www.youtube.com/watch?v=idpOunnpKTo
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
def Deriv_Lorenz(pt, t0, sigma=10, beta=8/3, rho=28):
    """Compute the time-derivative of a Lorentz system."""
    """
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    sigma = 10, beta = 8/3, rho = 28
    GenerationLimits = [(-15, 15), (-15, 15), (-15, 15)]
    plotLims = [(-30, 30), (-30, 30), (0, 50)]
    """
    x, y, z = pt

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return [dx, dy, dz]

def Deriv_Aizawa(pt, t0, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    """Compute the time-derivative of a Aizawa system."""
    """
    dx = (z-b) * x - d*y
    dy = d * x + (z-b) * y
    dz = c + a*z - z**3 /3 - x**2 + f * z * x**3

    a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.1
    GenerationLimits = [(-0.1, -0.1), (-0.1, 0.1), (-0.1, 0.1)]
    plotLims = [(-0.2, 0.2), (-0.2, 0.2), (0, 2)]
    """
    x, y, z = pt

    dx = ((z-b)*x) - d*y
    dy = d*x + ((z-b)*y)
    dz = c + a*z - ((z**3)/3) - (x**2) + f*z*(x**3)

    return [dx, dy, dz]

def Deriv_NewtonLeipnik(pt, t0, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    """Compute the time-derivative of a Aizawa system."""
    """
    dx = (z-b) * x - d*y
    dy = d * x + (z-b) * y
    dz = c + a*z - z**3 /3 - x**2 + f * z * x**3

    a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.1
    """
    x, y, z = pt

    dx = ((z-b)*x) - d*y
    dy = d*x + ((z-b)*y)
    dz = c + a*z - ((z**3)/3) - (x**2) + f*z*(x**3)

    return [dx, dy, dz]

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
def InitChaosAnimation():
    global Lines, Pts, x_t, fig, ax
    for line, pt in zip(Lines, Pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return Lines + Pts

# animation function.  This will be called sequentially with the frame number
def UpdateChaosAnimation(i):
    global Lines, Pts, x_t, fig, ax, speedUpFactor, rotationSpeed
    print(i, "done")
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (speedUpFactor * i) % x_t.shape[1]

    for line, pt, xi in zip(Lines, Pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(30, 0.3 * i*rotationSpeed)
    fig.canvas.draw()

    return Lines + Pts

def AnimateChaos(AttractorFunc, N_trajectories, GeneratorFunc, timeInterval=[0, 4], plotLims=[(-25, 25), (-35, 35), (5, 55)], frames=500, frame_interval=30, plotData=True, saveData={"save": False}):
    global Lines, Pts, x_t, fig, ax, speedUpFactor
    # Choose random starting points, uniformly distributed from -15 to 15
    startPoints = GeneratorFunc(N_trajectories)
    N_trajectories = startPoints.shape[0]

    # Solve for the trajectories
    time = np.linspace(timeInterval[0], timeInterval[1], frames*speedUpFactor)
    x_t = np.asarray([integrate.odeint(AttractorFunc, sP, time) for sP in tqdm(startPoints)])

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
    ax.set_xlim(plotLims[0])
    ax.set_ylim(plotLims[1])
    ax.set_zlim(plotLims[2])

    # Set point-of-view: specified by (altitude degrees, azimuth degrees)
    ax.view_init(30, 0)

    # Animate
    # InitAnim = functools.partial(InitChaosAnimation, Lines, Pts)
    # UpdateAnim = functools.partial(UpdateChaosAnimation, Lines=Lines, Pts=Pts, x_t=x_t, ax=ax, fig=fig)
    InitAnim = InitChaosAnimation
    UpdateAnim = UpdateChaosAnimation
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
N_trajectories = 27
AttractorFunc = Deriv_Aizawa
GeneratorFunc = GeneratePoints_UniformRandom
timeInterval = [0, 10]

GenerationLimits = [(-0.01, -0.01), (-0.01, 0.01), (-0.01, 0.01)]
plotLims = [(-1.5, 1.5), (-1.5, 1.5), (-0.5, 1.5)]
frames = 250
frame_interval = 30
speedUpFactor = 2
rotationSpeed = 3

plotData = False
saveData = {"save": True, "path":"AlgoVis/GeneratedVisualisations/AizawaAttractor_Random.gif", "fps": 30}
# Params

# RunCode
AnimateChaos(AttractorFunc, N_trajectories, functools.partial(GeneratorFunc, Limits=GenerationLimits), timeInterval=timeInterval, plotLims=plotLims, frames=frames, frame_interval=frame_interval, plotData=plotData, saveData=saveData)