"""Scientific Computation Project 3, part 2
Your CID here 01792931
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy
import time
from scipy.spatial.distance import pdist


def microbes(phi,kappa,mu,L = 1024,Nx=1024,Nt=1201,T=600,display=False):
    """
    Question 2.2
    Simulate microbe competition model

    Input:
    phi,kappa,mu: model parameters
    Nx: Number of grid points in x
    Nt: Number of time steps
    T: Timespan for simulation is [0,T]
    Display: Function creates contour plot of f when true

    Output:
    f,g: Nt x Nx arrays containing solution
    """

    #generate grid
    L = 1024
    x = np.linspace(0,L,Nx)
    dx = x[1]-x[0]
    dx2inv = 1/dx**2

    def RHS(y,t,k,r,phi,dx2inv):
        #RHS of model equations used by odeint

        n = y.size//2

        f = y[:n]
        g = y[n:]

        #Compute 2nd derivatives
        d2f = (f[2:]-2*f[1:-1]+f[:-2])*dx2inv
        d2g = (g[2:]-2*g[1:-1]+g[:-2])*dx2inv

        #Construct RHS
        R = f/(f+phi)
        dfdt = d2f + f[1:-1]*(1-f[1:-1])- R[1:-1]*g[1:-1]
        dgdt = d2g - r*k*g[1:-1] + k*R[1:-1]*g[1:-1]
        dy = np.zeros(2*n)
        dy[1:n-1] = dfdt
        dy[n+1:-1] = dgdt

        #Enforce boundary conditions
        a1,a2 = -4/3,-1/3
        dy[0] = a1*dy[1]+a2*dy[2]
        dy[n-1] = a1*dy[n-2]+a2*dy[n-3]
        dy[n] = a1*dy[n+1]+a2*dy[n+2]
        dy[-1] = a1*dy[-2]+a2*dy[-3]

        return dy


    #Steady states
    rho = mu/kappa
    F = rho*phi/(1-rho)
    G = (1-F)*(F+phi)
    y0 = np.zeros(2*Nx) #initialize signal
    y0[:Nx] = F
    y0[Nx:] = G + 0.01*np.cos(10*np.pi/L*x) + 0.01*np.cos(20*np.pi/L*x)

    t = np.linspace(0,T,Nt)

    #compute solution
    
    y = odeint(RHS,y0,t,args=(kappa,rho,phi,dx2inv),rtol=1e-6,atol=1e-6)
    f = y[:,:Nx]
    g = y[:,Nx:]
    
    if display:
        plt.figure()
        plt.contour(x,t,f)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of f')


    return f,g


def new_diff(f,h):
        #Coefficients for compact fd scheme
    alpha = 9/38
    a = (696-1191*alpha)/428 
    b = (2454*alpha-294)/535
    c = (1179*alpha-344)/2140
    #Compute LHF
    n = len(f)
    A = scipy.sparse.diags([alpha,1,alpha],[-1,0,1],shape=(n,n),format='csr')

    #Boundary 
    A[0,0],A[0,1] = 1,10 
    A[n-1,n-2],A[n-1,n-1]=10,1

    #Compute the RHS
    B = scipy.sparse.diags([c/9,b/4,a,-2*(c/9+b/4+a),a,b/4,c/9],[-3,-2,-1,0,1,2,3],shape=(n,n),format='csr')
    B = B.tolil()
    B[0,0],B[0,1],B[0,2],B[0,3],B[0,4] = 145/12,-76/3,29/2,-4/3,1/12
    B[1,n-2],B[1,n-1] =c/9,b/4
    B[2,n-1] = c/9
    B[n-3,0]= c/9
    B[n-2,0],B[n-2,1]= b/4,c/9
    B[n-1,n-5],B[n-1,n-4],B[n-1,n-3],B[n-1,n-2],B[n-1,n-1] = 1/12,-4/3,29/2,-76/3,145/12

    b_vector = B*f*h

    #Compute 2nd derivatives
    d2f = scipy.sparse.linalg.spsolve(A,b_vector)

    return d2f


def analyzefd():
    """
    Question 2.1 ii)
    Add input/output as needed

    """
    def func (x):
        return(np.sin(x))
    
    def func2 (x):
        return (-np.sin(x))
    
    xx = np.linspace(0,2*np.pi,100,endpoint=False)
    dx = xx[1]-xx[0]
    f = func(xx)
    
    exact_solution = func2(xx)
    t1 = time.time()
    for j in range(1000):
        compact_f = new_diff(f,1/dx**2)
    t2 = time.time()
    t3 = time.time()
    for j in range(1000):
        central_f = (f[2:]-2*f[1:-1]+f[:-2])/dx**2
    t4 = time.time()
    
    print('Iteration time of 1000 run for compact scheme = {}'.format(t2-t1))
    print('Iteration time of 1000 run for central difference scheme = {}'.format(t4-t3))
    
    plt.semilogy(xx[1:-1],abs(exact_solution[1:-1]-compact_f[1:-1]),label = 'Compact scheme')
    plt.semilogy(xx[1:-1],abs(exact_solution[1:-1]-central_f),label = 'Central difference scheme')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('$X$')
    plt.ylabel('$Error$')
    plt.title('Log error plot for one period')
    plt.show()
    return None #modify as needed


def dynamics():
    """
    Question 2.2
    Add input/output as needed

    """
    #Solution plot
    kappa = 1.5
    plt.figure()
    f_new,g_new = microbes(phi=0.3,kappa=kappa,mu=0.4*kappa,L=1024,Nx=1024,Nt=1024,T=500,display=False)
    plt.plot(np.arange(100, 1024), f_new[100:,-1],label='f solution')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Solution plot')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    eps = np.linspace(0,10,80)
    kappa_list =[1.5,1.7,2]
    #Plot fractal dimension for f and g
    for kappa in kappa_list: 
        phi,mu = 0.3,0.4*kappa
        L = 1024
        Nx= 1024
        Nt = 1024
        T=500
        display = False
    
        c = np.zeros(len(eps))
        c2 = np.zeros(len(eps))
        f,g = microbes(phi,kappa,mu,L,Nx,Nt,T,display)
        #fractal dimension for f
        y = f[len(f)//2:]
        y1 = y[:-1:2]
        y2 = y[1::2]
        A = np.vstack([y1,y2]).T
        D = pdist(A)

        for i in range(len(eps)):
            c[i]=sum(np.heaviside(eps[i]-D,1))
        #Fractal dimension for g
        y = g[len(g)//2:]
        y1 = y[:-1:2]
        y2 = y[1::2]
        A = np.vstack([y1,y2]).T
        D = pdist(A)

        for i in range(len(eps)):
            c2[i]=sum(np.heaviside(eps[i]-D,1)) 
    
    
        slope = np.polyfit(np.log(eps[10:-20]),np.log(c[10:-20]),1)
        slope2 = np.polyfit(np.log(eps[10:-20]),np.log(c2[10:-20]),1)
        
        plt.figure()
        plt.loglog(eps,c,'x-',label = 'f')
        plt.loglog(eps[10:-20],c[10:-20],label = 'Least square fit slope ={}'.format(slope[0]))
        plt.loglog(eps,c2,'x-',label = 'g')
        plt.loglog(eps[10:-20],c2[10:-20],label = 'Least square fit slope ={}'.format(slope2[0]))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('$\epsilon$')
        plt.ylabel('C($\epsilon$)')
        plt.title('Fractal dimension for $\kappa$ ={}'.format(kappa))
        
    #Plot fractal dimension for data 3
    
    D3 = np.load('data3.npy')
    eps = np.linspace(10,100,100)
    c = np.zeros(len(eps))
    
    a = D3[:,:,2]
    y = a[a.shape[0]//2:]
    y1 = y[:-1:2]
    y2 = y[1::2]
    A = np.vstack([y1,y2]).T
    D = pdist(A)
    n = A.shape[1]
    for i in range(len(eps)):
        c[i]=D[D<eps[i]].size
    
    slope = np.polyfit(np.log(eps[10:-75]),np.log(c[10:-75]),1)

    plt.figure()
    plt.loglog(eps,c,'x-',label = 'C($\epsilon$)')
    plt.loglog(eps[10:-75],c[10:-75],label = 'Least square fit slope ={}'.format(slope[0]))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('$\epsilon$')
    plt.ylabel('C($\epsilon$)')
    plt.title('Fractal dimension for data3')
    return None #modify as needed

if __name__=='__main__':
    analyzefd()
    dynamics()
