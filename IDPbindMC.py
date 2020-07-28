import sys
import numpy as np
from scipy.special import softmax
import seq_list as sl
import ChargedGaussian as cg
import multiprocessing as mp

# Command: python IDP-IDP_GaussChain_MC [seqname1] [seqname2]



write_configs_to_file = False # write generated chain configurations to files or not

T    = 1    # T = l/l_B = reduced temperature
rD   = 3    # Debye screening length

Rmax    = 100  # Maximum distance between two chains
n_R     = 100  # number of different R

n_ch1s, n_ch2s = int(1e2), int(1e2) # number of chain configurations
n_pairs = n_ch1s*n_ch2s

# Import sequences
seqname1, seqname2 = sys.argv[1], sys.argv[2]  
sig1, N1, seq1 = sl.get_the_charge(seqname1)
sig2, N2, seq2 = sl.get_the_charge(seqname2)
 
# Generate chain configurations    
def PolyGen1(_):    
    return cg.ChargedPolymer(sig1)

def PolyGen2(_):
    return cg.ChargedPolymer(sig2)

pool = mp.Pool(processes=2) 
chain1s = pool.map(PolyGen1,range(n_ch1s))
chain2s = pool.map(PolyGen1,range(n_ch2s))
pool.close() 

# Store the configurations for possible later use
if write_configs_to_file:
    file1 = open( seqname1 + '_' + str(n_chain) + '_cogfigs.txt', 'a')
    file2 = open( seqname2 + '_' + str(n_chain) + '_cogfigs.txt', 'a')

    for i in range(n_chain):
        np.savetxt(file1, chain1s[i].XYZ.T)
        np.savetxt(file2, chain2s[i].XYZ.T)

    file1.close()
    file2.close()


# Boltzmann factor of a pair of chain configs with various R
Rlist = np.linspace(1, Rmax, n_R)
dR    = Rlist[1]-Rlist[0]


def Uintra_fixed_configs( ch ):
    Uhomo  = cg.Uel_intra( ch, rD=rD )
    return Uhomo


def Uinter_fixed_configs( ch_n ):
    
    ch1, ch2 = chain1s[ int(ch_n/n_ch2s) ], chain1s[ int(ch_n % n_ch2s) ]

    Uall = [] 

    for R in Rlist:
        
        R1 = (0,0,R)
        Uhetero = cg.Uel_inter( ch1, ch2, R1diff=R1, rD=rD )
        Uall.append( Uhetero )

    return Uall


pool = mp.Pool(processes=40) 
Uintra1 = pool.map( Uintra_fixed_configs, chain1s )
Uintra2 = pool.map( Uintra_fixed_configs, chain2s )
Uinters = pool.map( Uinter_fixed_configs, np.arange(n_pairs) )

pool.close() 

P1 = softmax( ( -np.array(Uintra1) +np.min(Uintra1) )/T )
P2 = softmax( ( -np.array(Uintra2) +np.min(Uintra2) )/T )



Q12_configs = np.exp( -np.array(Uinters)/T ).dot(Rlist*Rlist)/np.sum( Rlist*Rlist ) 

QinterV = P1.dot( Q12_configs.reshape((n_ch1s, n_ch2s)).dot(P2) )

B2V = 1 - QinterV 

print('B2V:', B2V)

    