# using this for theory: https://www.youtube.com/watch?v=_6bKJrGCuJk

import numpy
import matplotlib.pyplot as plt

# functions ------------------------------------------------------------

def hhcoil_flux_density(P,I,N,R):
    # calculate field strength for a helmholtz coil
    # position = position from the center of the coil (origin)
    # current = current running through the coil
    # loops = number of loops on each copper coil in the hh coil
    # radius = radius of each coil (equal)

    B = (mu_0*0.5*N*I*R**2)*(1/(((R)**2+(P+0.5*R)**2)**(3/2))+1/(((R)**2+(P-0.5*R)**2)**(3/2)))

    return B;

flux_density = []
field_stength = []

# params ------------------------------------------------------------

n = 500
mu_0 = 1.25663706e-6 
I = 1.5; N = 100; R = 50;

# main ------------------------------------------------------------

space = numpy.linspace(-R/2,+R/2,n)

for point in space:
    flux_density.append(hhcoil_flux_density(point,I,N,R))

for point in flux_density:
    field_stength.append(point/mu_0)

fig, axs = plt.subplots(2)
axs[0].set_title('Magnetic Flux Density in Helmholtz Coil')
axs[0].set_ylabel('Flux Density, B [kg*s^-2*A^-2]')
axs[0].plot(space,flux_density)
axs[1].set_title('Magnetic Field Strength in Helmholtz Coil')
axs[1].set_xlabel('Position Along Coil [m]')
axs[1].set_ylabel('Strength, H [A/m]')
axs[1].plot(space,field_stength)
plt.show()