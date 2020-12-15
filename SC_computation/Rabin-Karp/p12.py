""" Your college id here:01792931
    Template code for part 2, contains 3 functions:
    codonToAA: returns amino acid corresponding to input amino acid
    DNAtoAA: to be completed for part 2.1
    pairSearch: to be completed for part 2.2
"""


def codonToAA(codon):
	"""Return amino acid corresponding to input codon.
	Assumes valid codon has been provided as input
	"_" is returned for valid codons that do not
	correspond to amino acids.
	"""
	table = {
		'ATA':'i', 'ATC':'i', 'ATT':'i', 'ATG':'m',
		'ACA':'t', 'ACC':'t', 'ACG':'t', 'ACT':'t',
		'AAC':'n', 'AAT':'n', 'AAA':'k', 'AAG':'k',
		'AGC':'s', 'AGT':'s', 'AGA':'r', 'AGG':'r',
		'CTA':'l', 'CTC':'l', 'CTG':'l', 'CTT':'l',
		'CCA':'p', 'CCC':'p', 'CCG':'p', 'CCT':'p',
		'CAC':'h', 'CAT':'h', 'CAA':'q', 'CAG':'q',
		'CGA':'r', 'CGC':'r', 'CGG':'r', 'CGT':'r',
		'GTA':'v', 'GTC':'v', 'GTG':'v', 'GTT':'v',
		'GCA':'a', 'GCC':'a', 'GCG':'a', 'GCT':'a',
		'GAC':'d', 'GAT':'d', 'GAA':'e', 'GAG':'e',
		'GGA':'g', 'GGC':'g', 'GGG':'g', 'GGT':'g',
		'TCA':'s', 'TCC':'s', 'TCG':'s', 'TCT':'s',
		'TTC':'f', 'TTT':'f', 'TTA':'l', 'TTG':'l',
		'TAC':'y', 'TAT':'y', 'TAA':'_', 'TAG':'_',
		'TGC':'c', 'TGT':'c', 'TGA':'_', 'TGG':'w',
	}
	return table[codon]


def DNAtoAA(S):
    """Convert genetic sequence contained in input string, S,
    into string of amino acids corresponding to the distinct
    amino acids found in S and listed in the order that
    they appear in S
    """


    AA = []
    #Loop over string s 3 at a time
    for i in range(0,len(S),3):
        # check if the amino acids is already in AA, if not, add in.
        if codonToAA(S[i:i+3]) not in AA:
            AA.append(codonToAA(S[i:i+3]))

    AA = ''.join(AA)

    return AA

def convert_str(S):

    #Convert string to corresponding number
    c2b = {}
    c2b['A']=0
    c2b['C']=1
    c2b['G']=2
    c2b['T']=3
    
    for string, values in c2b.items():    
        if S == string:
            return(values)



def pairSearch(L,pairs):
    """Find locations within adjacent strings (contained in input list,L)
    that match k-mer pairs found in input list pairs. Each element of pairs
    is a 2-element tuple containing k-mer strings
    """
        
    q = 997
    locations = []
	#Loop over pairs
    for index_p,element_p in enumerate(pairs):

        #Three index we need to find ,  Loop over string set
        for index_l in range(len(L)-1):

            
        
            
            element_L1,element_L2=''.join(L[index_l]),''.join(L[index_l+1])
            
            n=len(element_L1)
            m=len(element_p[0])

            pair_1,pair_2=element_p[0],element_p[1]
            


            bm = (4**m) % q


            #Define initial hi
            hi_1 = 0
            hi_2 = 0
            for i in range(m):
                hi_1 += (convert_str(element_L1[i])*4**(m-i-1)) % q
                hi_2 += (convert_str(element_L2[i])*4**(m-i-1)) % q 


            #Define hp
            hp_1 = 0
            hp_2 = 0
            for i in range(m):
                hp_1 += (convert_str(pair_1[i])*4**(m-i-1)) % q
                hp_2 += (convert_str(pair_2[i])*4**(m-i-1)) % q

            for ind in range(1,n-m+1):
            #Update 
                hi_1 = (4*hi_1 - convert_str(element_L1[ind-1])*bm + convert_str(element_L1[ind-1+m])) % q
                hi_2 = (4*hi_2 - convert_str(element_L2[ind-1])*bm + convert_str(element_L2[ind-1+m])) % q

                if (hi_1==hp_1 and hi_2==hp_2):
					#Check if string matches
                    if (element_L1[ind:ind+m] == pair_1 ) and ( element_L2[ind:ind+m] == pair_2):
                           locations.append([ind,index_l,index_p])
                          
    return locations

if __name__=='__main__':
    
  S = 'ATAATCATAATG'
  print(DNAtoAA(S))
  

  
  #Read the text file 9 character at a time
  L=[]
  with open('test_sequence.txt') as f:
    while True:
        c = f.read(9)
        L.append([c])
        if not c:
            break
  L.pop(-1)
  
  pairs=(['TCG','GAT'],['CTG','AGT'])

  print( pairSearch(L,pairs))
  L2 = ['GCAATTCGT','TCGTTGATC']
  print(pairSearch(L2,pairs))
    


