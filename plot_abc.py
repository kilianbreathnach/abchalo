import numpy as np
import corner


thetas = np.loadtxt("thetas.dat")

fig = corner.corner()
