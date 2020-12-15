"""Scientific Computation Project 2, part 2
Your CID here:
"""
import numpy as np
import networkx as nx
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy as scipy


def rwgraph(G,i0=0,M=100,Nt=100):
    """ Question 2.1
    Simulate M Nt-step random walks on input graph, G, with all
    walkers starting at node i0
    Input:
        G: An undirected, unweighted NetworkX graph
        i0: intial node for all walks
        M: Number of walks
        Nt: Number of steps per walk
    Output: X: M x Nt+1 array containing the simulated trajectories
    """
    adj_list = nx.adjacency_matrix(G)
    #Find the degree for each node
    Degree = [i[1] for i in G.degree]

    #Get the nonzero values' (x,y) index in sparse matrix
    Nonzero_x,Nonzero_y = adj_list.nonzero()

    #Set up the probability sparse matrix to save memeory space
    Prob = scipy.sparse.lil_matrix(np.zeros(list(adj_list.shape)))
    Prob[Nonzero_x,Nonzero_y] = [adj_list[i,j]/Degree[i] for i,j in zip(Nonzero_x,Nonzero_y)]

    X = np.zeros((Nt+1,M))

    X[0,:] = i0

    #M Nt-step random walks X
    for i in range(Nt):
        X[i+1,:] = [np.random.choice(list(G.adj[j]),
                            1,p=[ Prob[int(j),i]  for i in list(G.adj[j]) ] ) for j in X[i,:]]

    return X


def rwgraph_analyze1(input=(None)):
    """Analyze simulated random walks on
    Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    G = nx.barabasi_albert_graph(2000,4,seed=0)
    adj_1 = nx.adjacency_matrix(G)
    Degree = [i[1] for i in G.degree]

    #Get the nonzero values' (x,y) index in sparse matrix
    Nonzero_x,Nonzero_y = adj_1.nonzero()

    #Set up the probability sparse matrix to save memeory space
    Prob = scipy.sparse.lil_matrix(np.zeros(list(adj_1.shape)))

    Prob[Nonzero_x,Nonzero_y] = [adj_1[i,j]/Degree[i] for i,j in zip(Nonzero_x,Nonzero_y)]

    Nt,M = 200,2000
 

    io = Degree.index(max(Degree))
    x = np.zeros((Nt+1,M))

    x[0,:] = io
    Result = [Degree[io]]

    #M Nt-step random walks 
    for ind in range(Nt):
        x[ind+1,:] = [ np.random.choice( list(G.adj[j]),1,p=[ Prob[int(j),i]  for i in list(G.adj[j]) ] ) for j in x[ind,:] ]
    #Save the average of degree for all nodes at time Nt
        Result.append(np.mean([ Degree[i] for i in list(map(int, x[ind+1,:]))]))
        
    #Calculate the distribution of walkers at end
    Distri = [np.count_nonzero(x[Nt,:] == i) for i in range(len(Degree))]
    
    #Display walk
    plt.figure(figsize=(10,5))
    plt.plot(Result,'x--')
    plt.axhline(y=max(Degree), color='b', linestyle='-.',label='Min')
    plt.axhline(y=np.mean(Degree), color='r', linestyle='-.',label='Average')
    plt.axhline(y=min(Degree), color='yellow', linestyle='-.',label='Min')
    plt.xlabel('Time')
    plt.ylabel('$Averge node degree$')
    plt.title("Average of all walkers node degree at each time" )
    plt.legend(['Averege degree each simulation','Max degree','Original averege degree','Min degree'],loc=5)
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(Distri,'-.')
    plt.plot(np.argmax(Degree),Distri[np.argmax(Degree)],'o')
    plt.xlabel('Nodes')
    plt.ylabel('Number of walkers')
    plt.title(" Distribution of walkers for each node at end" )
    plt.legend(['Number of walkers','Highest degree node'],loc=5)
    plt.show()


    return None #modify as needed


def rwgraph_analyze2(input=(None)):
    """Analyze similarities and differences
    between simulated random walks and linear diffusion on
    Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    G = nx.barabasi_albert_graph(2000,4,seed=0)
    A = nx.adjacency_matrix(G)
    Degree = [i[1] for i in G.degree]
    Nt=100
    io=np.argmax(Degree)
    Q = np.diag(A.toarray().sum(axis=1))
    L = Q-A.toarray()
    Ls = np.matmul(np.linalg.inv(Q),L)
    LsT = np.transpose(Ls)
    
    def Laplacian(y,t):
    
        dydt = np.matmul(L,-y)
        return dydt
    def Laplacian_s(y,t):
    
        dydt = np.matmul(Ls,-y)
        return dydt
    def Laplacian_sT(y,t):
    
        dydt = np.matmul(LsT,-y)
        return dydt
    N = G.number_of_nodes()
    iarray = np.zeros((N,Nt+1))
    tarray = np.linspace(0,10,Nt+1)
    
    A= nx.adjacency_matrix(G).toarray()
    iarray[io,0] = 500
    X_s = odeint(Laplacian_s,iarray[:,0],tarray)
    X_sT = odeint(Laplacian_sT,iarray[:,0],tarray)
    
    plt.figure(figsize=(10,5))
    plt.plot(np.arange(0,N), X_sT[Nt,:])
    plt.plot(np.argmax(Degree),X_sT[Nt,np.argmax(Degree)],'o')
    plt.xlabel('Nodes')
    plt.ylabel('Number of walkers')
    plt.title(" Distribution of walkers for each node at end with L_sT operator" )
    plt.legend(['Number of walkers','Highest degree node'],loc=5)
    plt.show()
    plt.show()

    return None #modify as needed



def modelA(G,x=0,i0=0.1,beta=1.0,gamma=1.0,tf=5,Nt=1000):
    """
    Question 2.2
    Simulate model A

    Input:
    G: Networkx graph
    x: node which is initially infected with i_x=i0
    i0: magnitude of initial condition
    beta,gamma: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    iarray: N x Nt+1 Array containing i across network nodes at
                each time step.
    """
    def RHS(y,t):
    
        dydt = -beta * y+gamma * (1-y) * np.einsum('ij,j->i', A,y)

        return dydt
    
    N = G.number_of_nodes()
    iarray = np.zeros((N,Nt+1))
    tarray = np.linspace(0,tf,Nt+1)
    
    A= nx.adjacency_matrix(G).toarray()
    iarray[i0,0] = 1
    X = odeint(RHS,iarray[:,0],tarray)

    plt.figure(figsize=(10,5))
    plt.plot(np.linspace(0,tf,Nt+1), X[:,np.argmax(Degree)],'>')
    plt.plot(np.linspace(0,tf,Nt+1), X[:,np.argmin(Degree)],'<',color='red')
    plt.plot(np.linspace(0,tf,Nt+1), X,'--',color='grey')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend(['Max degree node','Min degree node','All nodes'],loc=1)
    plt.title("Concentration for model A at each node in each time step" )
    plt.show()


    return iarray


def transport(input=(None)):
    """Analyze transport processes (model A, model B, linear diffusion)
    on Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    tf=100
    Nt=1000
    G = nx.barabasi_albert_graph(100, 5, seed=0)
    Degree = [i[1] for i in G.degree]
    i0 = np.argmax(Degree)
    i1 = np.argmin(Degree)
    def modelB(G,io,alpha,tf):
        def RHS(y,t):
            
            dydt[:N] = y[N:]
            dydt[N:] =  alpha * (np.einsum('ij,j->i', L,y[:N])-np.einsum('ij,i->i', L,y[:N]))
            
            return dydt
        
        N = G.number_of_nodes()
        iarray = np.zeros((N*2,Nt+1))
        tarray = np.linspace(0,tf,Nt+1)
        A= nx.adjacency_matrix(G).toarray()
        Q = A.sum(axis=1)
        L = np.diag(Q)-A
        iarray[io,0],iarray[N+io,0] = 1,1

        dydt = iarray[:,0]
        X = odeint(RHS,iarray[:,0],tarray)
        
               #Plot the graph
        plt.figure(figsize=(10,5))
        plt.plot(np.linspace(0,tf,Nt+1), X[:,N+np.argmax(Degree)],'bo')
        plt.plot(np.linspace(0,tf,Nt+1), X[:,N:],'--',color='grey',label='_nolegend_')
        plt.plot(np.linspace(0,tf,Nt+1), X[:,N+np.argmin(Degree)],'bo',color='red')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend(['Max degree node','Min degree node'],loc=1)
        plt.title("Concentration for model B for i at each time step" )
        plt.show()
        
    def Laplacian(G,io):
        
        def Lp(y,t):
            
            dydt = np.matmul(L,-y)
            return dydt
        N = G.number_of_nodes()
        iarray = np.zeros((N,Nt+1))
        tarray = np.linspace(0,20,Nt+1)
        
        A= nx.adjacency_matrix(G).toarray()
        Q = A.sum(axis=1)
        L = np.diag(Q)-A
        iarray[io,0] = 1
        
        X_1 = odeint(Lp,iarray[:,0],tarray)
        plt.figure(figsize=(10,5))
        plt.plot(np.linspace(0,tf,Nt+1), X_1)
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title("Concentration vs Time by Laplacian operator" )

        plt.show()





    alpha = -0.01
    Laplacian(G,i0)
    modelB(G, i0, alpha, tf)
    Laplacian(G,i1)
    modelB(G, i1, alpha, tf)
    modelA(G,x=0,i0=i1,beta=0.5,gamma=0.1,tf=tf,Nt=1000)
    N = G.number_of_nodes()
    
        

    return None #modify as needed







if __name__=='__main__':
    #add code here to call diffusion and generate figures equivalent
    #to those you are submitting
    G = nx.barabasi_albert_graph(10,2,seed=0)
    print(rwgraph(G,0))
    rwgraph_analyze1()
    rwgraph_analyze2()
    G = nx.barabasi_albert_graph(100, 5, seed=0)
    Degree = [i[1] for i in G.degree]
    modelA(G,x=0,i0=np.argmax(Degree),beta=0.5,gamma=0.1,tf=20,Nt=1000)
    transport()
    