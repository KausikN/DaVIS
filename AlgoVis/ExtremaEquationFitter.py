'''
Algorithm Visualisation for my approach to fitting a polynomial using extrema points
Link: 
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from Libraries import SeriesVisualiseLibrary as SVL

# Main Functions
# Algorithm Classes
class Polynomial:
    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.degree = len(coeffs) - 1

# Algorithm Functions
def ExtremaFitter_Fit(extremas):
    equation = Polynomial([0]*(len(extremas) + 2))

    return equation

# Driver Code
# Params
FitFunc = ExtremaFitter_Fit
# Params

# Converge for a value
# Params
extremas = [[0, 1], [1, -1], [2, 4]]

annotate = False
# Params

# RunCode
extremas = np.array(extremas)
plt.plot(extremas[:, 0], extremas[:, 1], color='#6699cc', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
plt.scatter(extremas[:, 0], extremas[:, 1])
plt.show()