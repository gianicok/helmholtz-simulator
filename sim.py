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

    return B*1e4; # gauss conversion

def hhcoil_elength(x,y):
    
    bound = 0.999
    max_y = max(y)

    e_x = []
    e_y = []

    for i in range(len(y)):
        if y[i] > bound*max_y:
            e_x.append(x[i])
            e_y.append(y[i])

    return(e_x,e_y)


# params ------------------------------------------------------------

flux_density = []

n = 500              # Number of points in sim
mu_0 = 1.25663706e-6 # μ_0, Permeability of Free Space [m*kg*s^-2*A^-2] 

I = 0.5; # I, Current [A]
N = 40; # N, Number of turns [turns]
R = 0.35; # R, Radius of single coil [m]

copper = 2*3.14159*R*N*2 # per helmholtz pair
print("Per Helmholtz Pair, ",copper)
print("Per Helmholtz Cage, ",3*copper)

# main ------------------------------------------------------------

space = numpy.linspace(-R/2,+R/2,n)
for point in space:
    flux_density.append(hhcoil_flux_density(point,I,N,R))

e_x,e_y = hhcoil_elength(space,flux_density)

plt.title('Magnetic Flux Density in Helmholtz Pair')
plt.ylabel('Magnetic Flux Density, B [Gauss]')
plt.plot(space,flux_density,color='red')
plt.plot(e_x,e_y,color='black')

plt.gcf().text(0.1,0.95,
               'I = '+str(I)+' A, N = '+str(N)+' turns, R = '+str(R)+' m, l_e ='+str((round(e_x[-1]-e_x[0],2))*100)+' cm',
                fontsize = 14, bbox = dict(facecolor = 'red', alpha = 0.5))

plt.show()