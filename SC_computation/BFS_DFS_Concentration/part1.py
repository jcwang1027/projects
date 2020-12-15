"""Scientific Computation Project 2, part 1
Your CID here:
"""

import networkx as nx

def flightLegs(Alist,start,dest):
    """
    Question 1.1
    Find the minimum number of flights required to travel between start and dest,
    and  determine the number of distinct routes between start and dest which
    require this minimum number of flights.
    Input:
        Alist: Adjacency list for airport network
        start, dest: starting and destination airports for journey.
        Airports are numbered from 0 to N-1 (inclusive) where N = len(Alist)
    Output:
        Flights: 2-element list containing:
        [the min number of flights between start and dest, the number of distinct
        jouneys with this min number]
        Return an empty list if no journey exist which connect start and dest
    """
    #Use BFS to find the shortest path length
    L2 = [0 for l in range(len(Alist))] #Labels
    L4 = [[] for l in range(len(Alist))] #Paths 
    Q=[]
    Q.append(start)
    L2[start]=1
    L4[start] = [start]

    while len(Q)>0:
        x = Q.pop(0) #remove node from front of queue
        for v in Alist[x]:
            if L2[v]==0:
                Q.append(v) #add unexplored neighbors to back of queue
                L2[v]=1
                L4[v].extend(L4[x]) #Add path to node x and node v to path
                L4[v].append(v)     #for node v           

    #Use DFS to find all shortest path from sourse to destination
    shortest_len = len(L4[dest])-1
    paths = []
   
    def find_path(neighbor,start,dest,shortest,path=[]): 
        path=path+[start] 
        if start==dest:
            paths.append(path) 
        for node in neighbor[start]:
            #By using recursive function, we look for all path that has length less than or equal to 
            #the shortest path found using BFS and see which goes to the destination
            if node not in path and len(path)<=shortest:
                find_path(neighbor,node,dest,shortest,path)
        return(paths)

    Path = find_path(Alist, start, dest, shortest_len)
    Flights = [shortest_len,len(Path)]

    return Flights


def safeJourney(Alist,start,dest):
    """
    Question 1.2 i)
    Find safest journey from station start to dest
    Input:
        Alist: List whose ith element contains list of 2-element tuples. The first element
        of the tuple is a station with a direct connection to station i, and the second element is
        the density for the connection.
    start, dest: starting and destination stations for journey.

    Output:
        SList: Two element list containing safest journey and safety factor for safest journey
    """
    #Seprate connection and density
    neighbor = [[] for i in range(len(Alist))]
    density = [[] for i in range(len(Alist))]
    for index,value in enumerate(Alist):
        for i,(x,y) in enumerate(value):

            neighbor[index].append(x)
            density[index].append(y)
            
    L2 = [0 for l in range(len(Alist))] #Labels
    L3 = [1000 for l in range(len(Alist))] #Distances
    L4 = [[] for l in range(len(Alist))] #Paths 

    
    L3[start]=0
    L4[start] = [start]
    Q = [start]
    while len(Q)>0:
        x = Q.pop(0) 
        L2[x] = -1
        for (v,den) in zip(neighbor[x],density[x]):
            if L2[v]==0:
                Q.append(v)
                if L3[v] > L3[x]+den:
                    L3[v]=L3[x]+den
                    L4[v] = []          #We have a shorter path
                    L4[v].extend(L4[x]) #Add path to node x and node v to path
                    L4[v].append(v)     #for node v    
         #If the destination has not been visited
        if L2[dest] == 0:
            #Add the least density vertex in unvisited list to queue
            Unvisited = [[i,L3[i]] for i in range(len(G2)) if L2[i] == 0 ]
            Q.append(min(Unvisited, key= lambda t:t[1])[0])
    if L4[dest] == []:
        SList = []

    else:
        SList = [L4[dest], L3[dest]] 

    return SList

def shortJourney(Alist,start,dest):
    """
    Question 1.2 ii)
    Find shortest journey from station start to dest. If multiple shortest journeys
    exist, select journey which goes through the smallest number of stations.
    Input:
        Alist: List whose ith element contains list of 2-element tuples. The first element
        of the tuple is a station with a direct connection to station i, and the second element is
        the time for the connection (rounded to the nearest minute).
    start, dest: starting and destination stations for journey.

    Output:
        SList: Two element list containing shortest journey and duration of shortest journey
    """


    #Seprate connection and density
    neighbor = [[] for i in range(len(Alist))]
    density = [[] for i in range(len(Alist))]
    for index,value in enumerate(Alist):
        for i,(x,y) in enumerate(value):

            neighbor[index].append(x)
            density[index].append(y)
            
    L2 = [0 for l in range(len(Alist))] #Labels
    L3 = [1000 for l in range(len(Alist))] #Distances
    L4 = [[] for l in range(len(Alist))] #Paths 

    L3[start]=0
    L4[start] = [start]
    Q = [start]
    while len(Q)>0:
        
        x = Q.pop(0) 
       
        L2[x] = -1
        for (v,den) in zip(neighbor[x],density[x]):
            if L2[v]==0:
                if L3[v] > L3[x]+den:
                    Q.append(v)
                    L3[v]=L3[x]+den
                    L4[v] = []          #We have a shorter path
                    L4[v].extend(L4[x]) #Add path to node x and node v to path
                    L4[v].append(v)     #for node v  
                # Check if there is a path with less stops but same time
                elif L3[v] == L3[x]+den and len(L4[v]) > len(L4[x])+1:
                    L4[v] = []          
                    L4[v].extend(L4[x]) #Add path to node x and node v to path
                    L4[v].append(v)     #for node v   
        #If the destination has not been visited
        if L2[dest] == 0:
             #Add the least time vertex in unvisited list to queue
            Unvisited = [[i,L3[i]] for i in range(len(G2)) if L2[i] == 0 ]
            Q.append(min(Unvisited, key= lambda t:t[1])[0])
    if L4[dest] == []:
        SList = []

    else:
        SList = [L4[dest], L3[dest]] 

    return SList


def cheapCycling(SList,Clist):
    """
    Question 1.3
    Find first and last stations for cheapest cycling trip
    Input:
        SList: list whose ith element contains cheapest fare for arrival at and
        return from the ith station (stored in a 2-element list or tuple)
        Clist: list whose ith element contains a list of stations which can be
        cycled to directly from station i
    Stations are numbered from 0 to N-1 with N = len(SList) = len(Clist)
    Output:
        stations: two-element list containing first and last stations of journey
    """
    #Use dfs to group nodes with connection to each other
    def dfs(G,s,L2,label):
        Q=[]
        Q.append(s)
        L2[s]=label

        while len(Q)>0:
            x = Q.pop()
            for v in G[x]:
                if L2[v]==-1:
                    Q.append(v)
                    L2[v]=label
        return L2

    def connect(Clist):
        Lconnect = [-1 for n in range(len(Clist))]
        label=0
        for i in range(len(Clist)):
            if Lconnect[i]==-1:
                Lconnect = dfs(Clist,i,Lconnect,label)
                label = label+1

        return Lconnect,label
    

    [Lconnect,label] = connect(Clist)

    #Group routs that connects with each other
    def find_ind(L, compo):
        return [i for i,x in enumerate(L) if x == compo]

        #Find minimum fare in each group 
    def find_fare(nodes,SList):
        Score,Min_component = [],[]
        
            
        for ind,Comp in enumerate(nodes):
            #Find all combination of in and out in each group
            Score.append([[(i,j),SList[i][0]+SList[j][1]] for i in Comp for j in Comp if j != i])
            if Score[:][ind] != []:
                Min_component.append(min(Score[:][ind], key = lambda t: t[1]))
            else:
                Min_component.append([])
        return Min_component

    Group =[find_ind(Lconnect, i) for i in range(label)]
    Min_component = find_fare(Group,SList)


    stations = [(),100000]
    #Find minimum route out of all group
    for i in range(label):
        if Min_component[i] != [] and stations[1] > Min_component[i][1]:
            stations = Min_component[i]
    
    return stations[0]





if __name__=='__main__':
    #add code here if/as desired
    G = nx.barabasi_albert_graph(100,20,seed=0)
    adj_list = []
    #Creat Adjacency list
    for node, adjacencies in enumerate(G.adjacency()):
        adj_list.append(list(adjacencies[1]))
    print(flightLegs(adj_list,0,5))

    G2 = [[(1,3),(2,5)],[(0,3),(3,9)],[(0,5),(3,2),(4,6)],[(1,9),(2,2),(4,3)],[(2,6),(3,3)],[(6,3)],[(5,3)]]
    print(safeJourney(G2,0,4))
    G2 = [[(1,3),(2,5)],[(0,3),(3,9)],[(0,5),(3,2),(4,6)],[(1,9),(2,2),(4,4)],[(2,6),(3,4)]]
    print(shortJourney(G2,0,4))

    Clist=[[1,3],[0,2],[1,3],[0,2],[5,7],[4,6],[5,7,8],[4,6,8],[6,7],[]]
    SList=[(1,3),(5,6),(2,7),(3,8),(2,5),(1,3),(9,2),(5,7),(2,9),(2,3)]
    print(cheapCycling(SList,Clist))
    L=None #modify as needed
