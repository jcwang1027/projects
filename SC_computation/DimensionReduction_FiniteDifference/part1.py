"""Scientific Computation Project 3, part 1
Your CID here 01792931
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import mode

def hfield(r,th,h,levels=50):
    """Displays height field stored in 2D array, h,
    using polar grid data stored in 1D arrays r and th.
    Modify as needed.
    """
    thg,rg = np.meshgrid(th,r)
    xg = rg*np.cos(thg)
    yg = rg*np.sin(thg)
    plt.figure()
    plt.contourf(xg,yg,h,levels)
    plt.axis('equal')
    plt.colorbar()
    return None

def repair1(R,p,l=1.0,niter=10,inputs=()):
    """
    Question 1.1: Repair corrupted data stored in input
    array, R.
    Input:
        R: 2-D data array (should be loaded from data1.npy)
        p: dimension parameter
        l: l2-regularization parameter
        niter: maximum number of iterations during optimization
        inputs: can be used to provide other input as needed
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
    """
    #problem setup
    R0 = R.copy()
    a,b = R.shape
    iK,jK = np.where(R0 != -1000) #indices for valid data
    aK,bK = np.where(R0 == -1000) #indices for missing data

    S = set()
    for i,j in zip(iK,jK):
            S.add((i,j))

    #Set initial A,B
    A = np.ones((a,p))
    B = np.ones((p,b))

    #Create lists of indices used during optimization
    mlist = [[] for i in range(a)]
    nlist = [[] for j in range(b)]

    for i,j in zip(iK,jK):
        mlist[i].append(j)
        nlist[j].append(i)

    dA = np.zeros(niter)
    dB = np.zeros(niter)

    for k in range(niter):
        Aold = A.copy()
        Bold = B.copy()

        #Loop through elements of A and B in different
        #order each optimization step
        for m in np.random.permutation(a):
            for n in np.random.permutation(b):
                if n < p: #Update A[m,n]
                    Bfac = 0.0
                    Asum = 0

                    for j in mlist[m]:
                        Bfac += B[n,j]**2
                        Rsum = 0
                        for k in range(p):
                            if k != n: Rsum += A[m,k]*B[k,j]
                        Asum += (R[m,j] - Rsum)*B[n,j]

                    A[m,n] = Asum/(Bfac+l) #New A[m,n]
                if m<p:
                    #Add code here to update B[m,n]
                    B[m,n]=None #modify
        dA[k] = np.sum(np.abs(A-Aold))
        dB[k] = np.sum(np.abs(B-Bold))
        if k%10==0: print("k,dA,dB=",k,dA[k],dB[k])


    return A,B


def repair2(R,p,l=1.0,niter=10,inputs=()):
    """
    Question 1.1: Repair corrupted data stored in input
    array, R. Efficient and complete version of repair1.
    Input:
        R: 2-D data array (should be loaded from data1.npy)
        p: dimension parameter
        l: l2-regularization parameter
        niter: maximum number of iterations during optimization
        inputs: can be used to provide other input as needed
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
        """
        #problem setup
    R0 = R.copy()
    a,b = R.shape
    iK,jK = np.where(R0 != -1000) #indices for valid data
    aK,bK = np.where(R0 == -1000) #indices for missing data

    S = set()
    for i,j in zip(iK,jK):
            S.add((i,j))

    #Set initial A,B
    A = np.ones((a,p))
    B = np.ones((p,b))

    #Create lists of indices used during optimization
    mlist = [[] for i in range(a)]
    nlist = [[] for j in range(b)]

    for i,j in zip(iK,jK):
        mlist[i].append(j)
        nlist[j].append(i)

    dA = np.zeros(niter)
    dB = np.zeros(niter)

    np.random.seed(1)
    for k in range(niter):
        Aold = A.copy()
        Bold = B.copy()

        #Loop through elements of A and B in different
        #order each optimization step
        for m in np.random.permutation(a):
            for n in np.random.permutation(b):
                if n < p: 
                    R_mj,B_nj = R[m,mlist[m]],B[n,mlist[m]]
                    Asum1 = np.einsum("k,kj->j",A[m,0:p],B[0:p,mlist[m]])-A[m,n]*B_nj
                    Bfac = np.sum(B_nj**2)+l
                    Asum = np.dot(R_mj, B_nj)-np.dot(Asum1, B_nj)
                    A[m,n] = Asum/Bfac

                if m<p:
                    R_in, A_im = R[nlist[n],n], A[nlist[n],m]
                    Bsum1 = np.einsum("ik,k->i",A[nlist[n],0:p],B[0:p,n])-B[m, n]*A_im
                    Afac = np.sum(A_im**2)+l
                    Bsum = np.dot(R_in,A_im)-np.dot(Bsum1,A_im)
                    B[m,n] = Bsum/Afac
                    
        dA[k] = np.sum(np.abs(A-Aold))
        dB[k] = np.sum(np.abs(B-Bold))
        if k%10==0: print("k,dA,dB=",k,dA[k],dB[k])

    return A,B


def outwave(r0):
    """
    Question 1.2i)
    Calculate outgoing wave solution at r=r0
    See code/comments below for futher details
        Input: r0, location at which to compute solution
        Output: B, wave equation solution at r=r0

    """
    A = np.load('data2.npy')
    r = np.load('r.npy')
    th = np.load('theta.npy')
    
    Nr,Ntheta,Nt = A.shape
    B = np.zeros((Ntheta,Nt))
    
    # extract h(r=1,th,t) = f (th,t)
    radius = r[r0]
    f = A[0,:,:]
    
    C = np.fft.fftshift(np.fft.fft2(f))/(Nt*Ntheta)
    
    n_space = np.arange(0,Nt)
    m_space = np.arange(0,Ntheta)
    n2 = np.linspace(-Nt/2,Nt/2,Nt+1)
    m2 = np.linspace(-Ntheta/2,Ntheta/2,Ntheta+1)
    c = np.zeros((Ntheta, Nt),dtype = complex)
    cc = np.zeros((Ntheta, Nt),dtype = complex)
    
    for m in m_space:
        for n in n_space:
            C_mn = C[m,n]
    
            if np.isnan(abs(scipy.special.hankel1(m2[m],n2[n]*8))) == False:
    
                c[m,n] = C_mn/scipy.special.hankel1(m2[m], n2[n]*8)
            
            if np.isnan(abs(scipy.special.hankel1(m2[m],n2[n]*radius*8))) == False:
                cc[m,n] = c[m,n]*scipy.special.hankel1(m2[m], n2[n]*radius*8)
    
    
    B = np.fft.ifft2(np.fft.ifftshift(cc)*(Nt*Ntheta)).real


    return B

def analyze1():
    """
    Question 1.2ii)
    Add input/output as needed

    """
    A = np.load('data3.npy')
    r = np.load('r.npy')
    th = np.load('theta.npy')
    
    Nr,Ntheta,Nt = A.shape
    np.random.seed(1)
    radius = np.random.choice(Nr,3)
    theta_list = [np.pi/4,3/4*np.pi,5/4*np.pi]
    
    for i in range(len(theta_list)):
        plt.figure()
        for j in range(len(radius)):
            plt.plot(np.arange(0,Nt),A[radius[j],int((theta_list[i]/(2*np.pi))*(Ntheta-1)),:],label = 'radius = {}'.format(r[radius[j]]))
            plt.xlabel('Time')
            plt.ylabel('Height')
            plt.title(r'Height dynamics for $\theta ={}$'.format(theta_list[i]))
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
    return None #modify as needed




def reduce(H,inputs=()):
    """
    Question 1.3: Construct one or more arrays from H
    that can be used by reconstruct
    Input:
        H: 3-D data array
        inputs: can be used to provide other input as needed
    Output:
        arrays: a tuple containing the arrays produced from H
    """

    
    Nr,Ntheta,Nt = H.shape
    Matrix_2D = np.zeros([Nt,Nr,Ntheta])
    
    for i in range(Nt):
        Matrix_2D[i] = H[:,:,i]
        
    principle = []
    #Find the number of principle component we want save
    for i in Matrix_2D:
        u, s, vT = np.linalg.svd(i)
        cum_var = np.cumsum(s**2/sum(s**2))
        principle.append(np.argmax(cum_var>0.99))
    
    plt.figure()
    plt.plot(np.arange(0,len(Matrix_2D)), principle)
    plt.title("Principle components requires for 99% variance explained")
    plt.xlabel("2D-Matrix number")
    plt.ylabel("Principle components")
    plt.show()
    
    Pc_ind = np.max(principle)
    
    #Perform PCA and save the data
    UT, G = np.zeros((Pc_ind*Nt,Nr)),np.zeros((Pc_ind*Nt,Ntheta))
    
    for ind,matrix in enumerate(Matrix_2D):
        u,s,vT = np.linalg.svd(matrix)
    
        uT = np.transpose(u[:,:Pc_ind])
        g = np.dot(uT,matrix)
        
        UT[Pc_ind*ind:Pc_ind*(ind+1),:] = uT
        G[Pc_ind*ind:Pc_ind*(ind+1),:] = g
    
    return (UT,G)



def reconstruct(arrays,inputs=()):
    """
    Question 1.3: Generate matrix with same shape as H (see reduce above)
    that has some meaningful correspondence to H
    Input:
        arrays: tuple generated by reduce
        inputs: can be used to provide other input as needed
    Output:
        Hnew: a numpy array with the same shape as H
    """
    UT,G = arrays
    
    Pc_ind = 32
    U = np.transpose(UT)
    Nr = U.shape[0]
    Ntheta = G.shape[1]
    Nt = int(len(UT)/Pc_ind)
    Hnew = np.zeros([Nr,Ntheta,Nt])
    
    for ind in range(Nt):
        Hnew[:, :, ind] = np.dot(U[:,Pc_ind*ind:Pc_ind*(ind+1)], G[Pc_ind*ind:Pc_ind*(ind+1), :])
    
    Save = (Hnew.size * Hnew.itemsize)-(U.size * U.itemsize + G.size * G.itemsize)
    print("%d bytes or %f percent Memory size saved" % (Save,Save/(Hnew.size * Hnew.itemsize)*100))

    return Hnew


if __name__=='__main__':
    r = np.load('r.npy')
    th = np.load('theta.npy')
    D1 = np.load('data1.npy')
    
    hfield(r,th,D1)
    plt.title('Unrepaired R')
    plt.show()
    plt.figure()
    A,B=repair2(D1,p=8)
    R = np.dot(A, B)
    hfield(r,th,R)
    plt.title("Repaired R")
    plt.show()
    
    analyze1()
    
    H = np.load('data3.npy')
    reconstruct(reduce(H))