# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:04:52 2023

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#I start first by defining an equation for the velocity u:
def velocity(x, nu):
    u = -9.0*nu/(2.0*x)
    return u

#I then set reasonable values for dx, nu,  and Ngrid to set up an array of x going from 0 to 2
dx = 0.02
Ngrid = int(2.0/dx+1.0)
nu=0.01

#I set up my arrays for x and u. 
#For u, I call on the velocity function previously defined. I also set up the boundary conditions for u.
x = np.linspace(0, 2, Ngrid)
u = np.append([0], velocity(x[1:], nu))
u[0] = -np.abs(u[1])
u[-1] = np.abs(u[-2])

#I set up my initial sharp Gaussian
f = norm.pdf(x, 1., 1E-1)

#I set up dt and Nsteps knowing that dt<dx/u
dt = np.min(np.abs(dx/u))/2
Nsteps = 5000

#I define the diffusion coefficient D and the beta parameter
D= 3.0*nu
beta = D*dt/dx**2

#I set up the plot and animation here
plt.ion()
fig, ax = plt.subplots(1,1)

plt.title("$\Sigma$(r, t) vs r")
plt.xlabel("r (as a fraction of r/R$_0$)")
plt.ylabel("$\Sigma$(r,t)")
ax.plot(x, f, 'k-')
plt1, = ax.plot(x, f, 'ro')

fig.canvas.draw()

#Here is where the numerical solving begins for each step within Nsteps
for count in range(Nsteps):
    #Here is the implicit method for the diffusion part
    A = np.eye(Ngrid)*(1.0+2.0*beta)+np.eye(Ngrid, k=1)*-beta+np.eye(Ngrid, k=-1)*-beta
    f = np.linalg.solve(A, f)

    #Here is the Lax-Friedrichs method for the advection part
    f[1:Ngrid-1] = 0.5*(f[2:]+f[:Ngrid-2])-0.5*u[1:Ngrid-1]*(dt/dx)*(f[2:]-f[:Ngrid-2])

    #I set up the boundary conditions here
    f[0] = f[1]
    f[-1] = f[-2]
    
    #I finish the setup for the animation here
    plt1.set_ydata(f)
    fig.canvas.draw()
    plt.pause(0.001)
