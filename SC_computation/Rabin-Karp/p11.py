""" Your college id here: 01792931
    Template code for part 1, contains 4 functions:
    newSort, merge: codes for part 1.1
    time_newSort: to be completed for part 1.1
    findTrough: to be completed for part 1.2
"""

import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

def newSort(X,k=0):
    """Given an unsorted list of integers, X,
        sort list and return sorted list
    """

    n = len(X)
    if n==1:
        return X
    elif n<=k:
        for i in range(n-1):
            ind_min = i
            for j in range(i+1,n):
                if X[j]<X[ind_min]:
                    ind_min = j
            X[i],X[ind_min] = X[ind_min],X[i]
        return X
    else:
        L = newSort(X[:n//2],k)
        R = newSort(X[n//2:],k)
        return merge(L,R)


def merge(L,R):
    """Merge 2 sorted lists provided as input
    into a single sorted list
    """
    M = [] #Merged list, initially empty
    indL,indR = 0,0 #start indices
    nL,nR = len(L),len(R)

    #Add one element to M per iteration until an entire sublist
    #has been added
    for i in range(nL+nR):
        if L[indL]<R[indR]:
            M.append(L[indL])
            indL = indL + 1
            if indL>=nL:
                M.extend(R[indR:])
                break
        else:
            M.append(R[indR])
            indR = indR + 1
            if indR>=nR:
                M.extend(L[indL:])
                break
    return M


def time_newSort(inputs=None):
    """Analyze performance of newSort
    Use variables inputs and outputs if/as needed
    """
    
    
    #Generate 1000 string sample 
    N = 1000
    X = list(np.random.randint(1,N,N))
    
    
    
    Time_all=dict((k_index,[]) for k_index in inputs)
    
    for k_index in inputs:
        
        for i in range(1,N+1):
            
            sum_time = 0
            #Return average of 5 iteration
            for j in range(5):
           
                t1 = time.time()
                S = newSort(X[:i],k_index)
                t2 = time.time()
                sum_time += t2-t1
            
            Time_all[k_index].append(sum_time/5)
        
        sns.lineplot(np.arange(0, N), Time_all[k_index], label='k_index={}'.format(k_index))
    plt.xlabel('Input size')
    plt.ylabel('Iteration time')
    plt.legend(loc="upper right")
    plt.title('Iteration time vs Input size ')
    plt.show(block=False)

    


def findTrough(L):
    """Find and return a location of a trough in L
    """
    #Define endpoint index
    left = 0
    right = len(L)-1
    
    while left < right:
        mid = (left+right) // 2
    
        if L[mid] >= L[mid + 1]:
             #Trough exist in right half
            left = mid + 1
            
        else:
            #Trough exist in left half
            right = mid 
    
    #Deal with edge cases and empty list
    if L == []:
        return -(N+1)
    
    if L == 0 or L==len(L)-1:
        return min(L[0],L[end])
    
    return L[left],left


if __name__=='__main__':
    
    inputs = [0,10,30,200,1000]
    time_newSort(inputs)
    
    
    N=20
    
    L = list(np.random.randint(1,2*N,N))
    print(L)
    [a,b] = findTrough(L)
    print('Trough value =',a,'location index=' ,b)
    
    