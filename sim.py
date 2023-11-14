# using this for theory: https://www.youtube.com/watch?v=_6bKJrGCuJk
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

# class ----------------------------------------------------------------

class hh_pair:

    def __init__(self, N, R, name):
        self.N = N
        self.R = R
        self.name = name

    def flux_calc(self):
        self.flux = []
        self.space = numpy.linspace(-(self.R)/2,+(self.R)/2,n)
        for point in self.space: 
            self.flux.append(hhcoil_flux_density(point,I,self.N,self.R))

        self.elen_x, self.elen_y = hhcoil_elength(self.space,self.flux)

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
    # find values within the helmholtz pairs that have a field no less than 0.999 of the max
    
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

n = 500              # Number of points in sim
I = 2;               # I, Current [A]
mu_0 = 1.25663706e-6 # μ_0, Permeability of Free Space [m*kg*s^-2*A^-2] 

x = hh_pair(40,0.35,'x')
y = hh_pair(42,0.37,'y')
z = hh_pair(44,0.39,'z')

pairs = [x,y,z]

# main ------------------------------------------------------------

coils = image.imread("./coil.png")
#plt.imshow(coils)
#plt.show()

x.flux_calc()
y.flux_calc()
z.flux_calc()

# same plot
fig, (ax) = plt.subplots(1)
fig.suptitle('Helmholtz Pair Characteristics')

ax.add_artist(AnnotationBbox(OffsetImage(coils, zoom = 1), (min(x.space), min(x.flux)+(max(x.flux)-(min(x.flux)))/3), frameon = False))
ax.add_artist(AnnotationBbox(OffsetImage(coils, zoom = 1), (max(x.space), min(x.flux)+(max(x.flux)-(min(x.flux)))/3), frameon = False))

ax.set_ylabel("Magnetic Field (Gauss)")
ax.set_xlabel("Position Along Helmholtz Pair (m)")
ax.plot(x.space,x.flux,color='red',label="X Pair, I = "+str(I)+" A, R = "+str(x.R)+" m, N = "+str(x.N))
ax.plot(x.elen_x,x.elen_y,color='black')
ax.plot(y.space,y.flux,color='green',label="Y Pair, I = "+str(I)+" A, R = "+str(y.R)+" m, N = "+str(y.N))
ax.plot(y.elen_x,y.elen_y,color='black')
ax.plot(z.space,z.flux,color='blue',label="Z Pair, I = "+str(I)+" A, R = "+str(z.R)+" m, N = "+str(z.N))
ax.plot(z.elen_x,z.elen_y,color='black')
uniform = " x="+str((round(x.elen_x[-1]-x.elen_x[0],2))*100)+" cm, y="+str((round(y.elen_x[-1]-y.elen_x[0],2))*100)+" cm, z="+str((round(z.elen_x[-1]-z.elen_x[0],2))*100)+" cm"

ax.legend()
ax.set_title("uniform regions (black): "+uniform)
#fig.text(0.1,0.95,uniform,fontsize = 14, bbox = dict(facecolor = 'red', alpha = 0.5))
plt.show()

'''
plt.title('Magnetic Flux Density in Helmholtz Pair')
plt.ylabel('Magnetic Flux Density, B [Gauss]')
plt.plot(space,flux_density,color='red')
plt.plot(e_x,e_y,color='black')

plt.gcf().text(0.1,0.95,
               'I = '+str(I)+' A, N = '+str(N)+' turns, R = '+str(R)+' m, l_e ='+str((round(e_x[-1]-e_x[0],2))*100)+' cm',
                fontsize = 14, bbox = dict(facecolor = 'red', alpha = 0.5))

plt.show()'''