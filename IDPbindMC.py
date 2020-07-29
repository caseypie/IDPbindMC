import sys
import numpy as np
#from scipy.special import softmax
import seq_list as sl
import ChargedGaussian as cg
import multiprocessing as mp
import time

# Command: python IDP-IDP_GaussChain_MC [seqname1] [seqname2]


def softmax(v):
    A = np.exp(v - np.max(v))
    return A/np.sum(A, axis=0)


write_configs_to_file = False # write generated chain configurations to files or not

T    = 1.    # T = l/l_B = reduced temperature
rD   = 3.    # Debye screening length
cut  = 0.    # short-range cutoff for monomer-monomer interaction

Rmax    = 100  # Maximum distance between two chains
n_R     = 100  # number of different R

n_ch1s, n_ch2s = int(1e4), int(1e4) # number of chain configurations
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

t = time.perf_counter()

pool = mp.Pool(processes=40) 
chain1s = pool.map(PolyGen1,range(n_ch1s))
chain2s = pool.map(PolyGen2,range(n_ch2s))
pool.close() 

# Store the configurations for possible later use
if write_configs_to_file:
    fname1 = seqname1 + '_' + str(n_ch1s) + '_cogfigs.txt'
    fname2 = seqname2 + '_' + str(n_ch2s) + '_cogfigs.txt'

    file1 = open(fname1, 'w')
    file1.close()
    file1 = open(fname1, 'a')
    for i in range(n_ch1s):
        np.savetxt(file1, chain1s[i].XYZ.T)
    file1.close()


    file2 = open(fname2, 'w')
    file2.close()
    file2 = open(fname2, 'a')
    for i in range(n_ch2s):
        np.savetxt(file2, chain2s[i].XYZ.T)
    file2.close()

    print('Chain configurations generated; ' + str(time.perf_counter()-t) + ' seconds spent.', flush=True)

print('Chain configurations generated', flush=True)

# Boltzmann factor of a pair of chain configs with various R
Rlist = np.linspace(1, Rmax, n_R)
dR    = Rlist[1]-Rlist[0]


def Uintra_fixed_configs( ch ):
    Uhomo  = cg.Uel_intra( ch, rD=rD )
    return Uhomo

pool = mp.Pool(processes=40) 
Uintra1 = pool.map( Uintra_fixed_configs, chain1s )
Uintra2 = pool.map( Uintra_fixed_configs, chain2s )
pool.close()

print( 'Uintra calculation finished!', flush=True  )

def Qinter_fixed_configs( ch_n ):
    if ch_n % 1e3 == 0:
        print( ch_n, flush=True )
    
    ch1, ch2 = chain1s[ int(ch_n/n_ch2s) ], chain2s[ int(ch_n % n_ch2s) ]

    Uall = [] 

    for R in Rlist:
        
        R1 = (0,0,R)
        Uhetero = cg.Uel_inter( ch1, ch2, R1diff=R1, rD=rD )
        Uall.append( Uhetero )


    Q12_config_fixed = np.exp( -np.array(Uall)/T ).dot(Rlist*Rlist)/np.sum( Rlist*Rlist ) 
   
    return Q12_config_fixed

pool = mp.Pool(processes=40) 
Q12_configs = pool.map( Qinter_fixed_configs, np.arange(n_pairs) )
pool.close() 

P1 = softmax( ( -np.array(Uintra1) +np.min(Uintra1) )/T )
P2 = softmax( ( -np.array(Uintra2) +np.min(Uintra2) )/T )

QinterV = P1.dot( Q12_configs.reshape((n_ch1s, n_ch2s)).dot(P2) )

B2V = 1 - QinterV 

print('B2V:', B2V)

f = open( 'B2V_' + seqname1 + '_' + seqname2 + '_T' + str(T) + '_rD' + str(rD) + '_cut' + str(cut) + '.txt', 'w') 
f.write('B2V:' + str(B2V) )
f.close() 
