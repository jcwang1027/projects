"""Scientific Computation Project 4
Your CID here: 01792931
"""
import numpy as np
import networkx as nx
from scipy.integrate import odeint
from scipy.special import expit #sigmoid function
import matplotlib.pyplot as plt
import pandas as pd
from networkx.algorithms import centrality
from networkx.algorithms.community import greedy_modularity_communities

def data(fname='project4.csv'):
    """The function will load human mobility data from the input file and
    convert it into a weighted undirected NetworkX Graph.
    Each node corresponds to a country represented by its 3-letter
    ISO 3166-1 alpha-3  code. Each edge between a pair of countries is
    weighted with the number of average daily trips between the two countries.
    The dataset contains annual trips for varying numbers of years, and the daily
    average is computed and stored below.
    """

    df = pd.read_csv(fname,header=0) #Read dataset into Pandas dataframe, may take 1-2 minutes


    #Convert dataframe into D, a dictionary of dictionaries
    #Each key is a country, and the corresponding value is
    #a dictionary which has a linked country as a key
    #and a 2-element list as a value.  The 2-element list contains
    #the total number of trips between two countries and the number of years
    #over which these trips were taken
    D = {}
    for index, row in df.iterrows():
         c1,c2,yr,N = row[0],row[1],row[2],row[3]
         if len(c1)<=3:
             if c1 not in D:
                 D[c1] = {c2:[N,1]}
             else:
                 if c2 not in D[c1]:
                     D[c1][c2] = [N,1]
                 else:
                     Nold,count = D[c1][c2]
                     D[c1][c2] = [N+Nold,count+1]


    #Create new dictionary of dictionaries which contains the average daily
    #number of trips between two countries rather than the 2-element lists
    #stored in D
    Dnew = {}
    for k,v in D.items():
        Dnew[k]={}
        for k2,v2 in v.items():
            if v2[1]>0:
                v3 = D[k2][k]
                w_ave = (v2[0]+v3[0])/(730*v2[1])
                if w_ave>0: Dnew[k][k2] = {'weight':w_ave}

    G = nx.from_dict_of_dicts(Dnew) #Create NetworkX graph

    return G


def network(G,inputs=()):
    """
    Analyze input networkX graph, G
    Use inputs to provide any other needed information.
    """
    Num_com = list(greedy_modularity_communities(G))
    
    pos = nx.spring_layout(G)
    #Switch frozen set to dictionary
    y = 0
    Community = {}
    for v in Num_com:
      for x in list(v):
        Community.update({x:y})
      y += 1
    Community = dict(sorted(Community.items()))
    
    plt.figure(figsize=(10, 10)) 
    nx.draw(G, pos, node_size=20, cmap=plt.cm.RdYlBu, node_color=list(Community.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show()
        
    #Plot node degree distribution
    degrees = [d for i, d in G.degree()]

    ax1 = plt.hist(degrees,max(degrees))
    plt.xlabel('Degree')
    plt.ylabel('Total number of countries')
    plt.title(' Degree distribution')
    plt.show()

    #Compute degree and pagerank centrality measure
    degrees = centrality.degree_centrality(G)
    pagerank = nx.pagerank(G)

    #Combine the three measure data into dataframe
    Data = {'Degree': degrees,  'Pagerank': pagerank}
    Df = pd.DataFrame(Data, columns= ['Degree', 'Pagerank'])

    #Rank the three centrality measures and combine them to dataframe
    Rank_degrees = [key for (key, value) in sorted(degrees.items(), key=lambda item: item[1],reverse=True)]
    Rank_pagerank = [key for (key, value) in sorted(pagerank.items(), key=lambda item: item[1],reverse=True)]

    Data = {'Degree': Rank_degrees,'Pagerank': Rank_pagerank}
    Ranking = pd.DataFrame(Data, columns= ['Degree',  'Pagerank'])
    print(Ranking.head(10))
    
    

    return None




def modelBH(G,x=0,i0=0.1,alpha=0.45,beta=0.3,gamma=1e-3,eps=1.0e-6,eta=8,tf=20,Nt=1000):
    """
    Simulate model Brockmann & Helbing SIR model

    Input:
    G: Weighted undirected Networkx graph
    x: node which is initially infected with j_x=j0
    j0: magnitude of initial condition
    alpha,beta,gamma,eps,eta: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    tarray: size Nt+1 array
    jarray: Nt+1 x N array containing j across the N network nodes at
                each time step.
    sarray: Nt+1 x N array containing s across network nodes at each time step
    """

    N = G.number_of_nodes()
    tarray = np.linspace(0,tf,Nt+1)  
    sig = lambda x: x**eta/(1+x**eta) 
    init = np.zeros(2*N)
    init[N:] = np.ones(N)
    init[x] = i0
    init[N+x] = 1-i0
    F = nx.adjacency_matrix(G).toarray()
    P = F/np.sum(F,axis=0)
    
    def RHS(i, t):
        j, s = i[:N], i[N:]
        di = np.zeros(2*N)
        di[:N] = alpha*np.dot(np.dot(s,j),sig(j/eps)) -beta*j + gamma*(np.einsum("mn,m->n",P,j) - np.einsum("mn,n->n",P,j))
        di[N:] = -alpha*np.dot(np.dot(s,j),sig(j/eps)) + gamma*(np.einsum("mn,m->n",P,s) - np.einsum("mn,n->n",P,s))
        return di
    

    sol = odeint(RHS, init, tarray)
    jarray = sol[:,:N].T
    sarray = sol[:,N:].T
    return tarray, jarray, sarray



    
def analyze(G,inputs=()):
    """Compute effective distance matrix and
    analyze simulation results
    Input:
        G: Weighted undirected NetworkX graphs
        inputs: can be used to provide additional needed information
    Output:
        D: N x N effective distance matrix (a numpy array)

    """
    df = pd.read_csv('project4.csv',header=0) 
    N = G.number_of_nodes()
    F = nx.adjacency_matrix(G).toarray()
    P = F / np.sum(F, axis=0)
    d = (1 - np.log(P))
    #Find the shortest path with dijkstra algorithm
    def shortest_path(G, start):
        nodes = len(G)
        Visit = [0 for i in range(nodes)]
        D_mn = [100000 for i in range(nodes)]
        D_mn[start] = 0

        for ind in range(nodes):
            Dist = 100000
            for i in range(nodes):
                if Visit[i] == 0 and D_mn[i] < Dist:
                    new_ind = i
                    Dist = D_mn[i]
            Visit[new_ind] = 1
            for j in range(nodes):
                if Visit[j] == 0 and G[new_ind][j] != 0 and D_mn[j] > D_mn[new_ind] + G[new_ind][j]:
                    D_mn[j] = G[new_ind][j] + D_mn[new_ind]

        return D_mn
    D = np.array([shortest_path(d,i) for i in range(N)])
    
    Country_ind = df['reporting country'].duplicated() == False
    Country_ind = df['reporting country'][Country_ind].reset_index(drop=True)
    OL = [0,6,100]
    for i in OL:
        tarray,jarray,sarray = modelBH(G,x=i)

        #Find arrival time
        Time = (jarray >= 1e-4).argmax(axis=1)
        ind = np.append([i],np.nonzero(Time))
        x = D[i,ind]
        y = tarray[Time[ind]]
        x_space = np.linspace(np.min(x),np.max(x),1000)
        plt.figure()
        b,a = np.polyfit(x,y,1)
        plt.plot(x_space,b*x_space+a,'--',label='fitted line')
        plt.scatter(x, y,label='Country')
        plt.xlabel('$D_{eff}$')
        plt.ylabel(r'$T_{\alpha}$')
        plt.title('Arrival time vs Effective distance with outbreak location {}'.format(Country_ind[i]))
        plt.legend()
        plt.show()
        
        
    OL = np.arange(Country_ind.size)
    for OL_ind in [0,6,10]:
        Corelation = []
        for i in OL[:20]:
            tarray,jarray,sarray = modelBH(G,x=OL_ind)
            #Find arrival time
            Time = (jarray >= 1e-4).argmax(axis=1)
            ind = np.append([i],np.nonzero(Time))
            x = D[i,ind]
            y = tarray[Time[ind]]
            b, a = np.polyfit(x,y,1)
            Corelation.append(b)
        plt.figure()
        plt.plot(OL[:20],Corelation)
        plt.plot(OL_ind,Corelation[OL_ind],'ro',label='True outbreak location')
        plt.xlabel('Country index')
        plt.ylabel('Corelation')
        plt.title('Corelation of arrival time and effective distance for outbreak location:{}'.format(Country_ind[OL_ind]))
        plt.legend()
        plt.show()

    return D
        



if __name__=='__main__':
    #Add code below to call network and analyze so that they generate the figures
    #in your report.
    G = data()
    network(G)
    analyze(G)