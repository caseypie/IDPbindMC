# Jul 29, 2020
# MPI version of MC for IDP-IDP binding calculation

import sys
import numpy as np
#from scipy.special import softmax
import seq_list as sl
import ChargedGaussian as cg

import multiprocessing as mp
from mpi4py import MPI
import time

# Command: mpirun -n [n_core] python3 IDPbindMC_MPI [seqname1] [seqname2]

# n_core has to be consistent with the [n_core] in command line
n_core = 800


def softmax(v):
    A = np.exp(v - np.max(v))
    return A/np.sum(A, axis=0)

write_configs_to_file = True # write generated chain configurations to files or not

T    = 1.    # T = l/l_B = reduced temperature
rD   = 3.    # Debye screening length
cut  = 0.    # short-range cutoff for monomer-monomer interaction

Rmax    = 100  # Maximum distance between two chains
n_R     = 100  # number of different R

n_ch1s, n_ch2s = int(1e4), int(1e4) # number of chain configurations
n_pairs = n_ch1s*n_ch2s

# Build MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if rank == 0:
    # Import sequences
    seqname1, seqname2 = sys.argv[1], sys.argv[2]  
    sig1, N1, seq1 = sl.get_the_charge(seqname1)
    sig2, N2, seq2 = sl.get_the_charge(seqname2)
 
    # Generate chain configurations in rank=0
    def PolyGen1(_): 
        np.random.seed()  
        poly  = cg.ChargedPolymer(sig1)
        U     = cg.Uel_intra( poly, rD=rD )
        return poly, U

    def PolyGen2(_):
        np.random.seed()
        poly  = cg.ChargedPolymer(sig2)
        U     = cg.Uel_intra( poly, rD=rD )
        return poly, U

    t = time.perf_counter()
 
    pool = mp.Pool(processes=40) 
    chain1s, U1 = zip( *pool.map(PolyGen1,range(n_ch1s)) )
    chain2s, U2 = zip( *pool.map(PolyGen2,range(n_ch2s)) )
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

        print('Chain configurations generated; ' \
               + str(time.perf_counter()-t) \
               + ' seconds spent.', flush=True)
    else:  
        print('Chain configurations generated!', flush=True)
    
    print( 'Uintra calculation finished!', flush=True  )

else:
    chain1s = None
    chain2s = None

# Broadcast chain configurations
chain1s = comm.bcast(chain1s, root=0)
chain2s = comm.bcast(chain2s, root=0)

# Scatter chain pairs
size = comm.Get_size()
chunksize = int(n_pairs/size)

if rank == 0:
    pairs_sendbuf = np.arange(n_pairs).reshape( size, chunksize  )
    pairs_sendbuf = pairs_sendbuf.astype('int')
else:
    pairs_sendbuf = None


pairs = comm.scatter(pairs_sendbuf, root=0)


# Mayer function of a pair of chain configs 
# integrated with respect to various R
Rlist = np.linspace(1, Rmax, n_R)
dR    = Rlist[1]-Rlist[0]

def Minter_fixed_configs( n12 ):
    if n12 % 1e3 == 0:
        print( n12, flush=True )
    ch1, ch2 = chain1s[ int( n12 / n_ch2s) ], chain2s[ int( n12 % n_ch2s) ]

    Uall = [] 

    for R in Rlist:        
        R1 = (0,0,R)
        U12 = cg.Uel_inter( ch1, ch2, R1diff=R1, rD=rD )
        Uall.append( U12 )

    M12_config_fixed = ( np.exp(-np.array(Uall)/T) - 1 ).dot(Rlist*Rlist)/np.sum( Rlist*Rlist ) 
   

    return M12_config_fixed

M12_configs_sendbuf = np.zeros(pairs.shape[0] )

for i, n_p  in enumerate(pairs):
    M12_configs_sendbuf[i] = Minter_fixed_configs( n_p )

#print(  M12_configs_sendbuf , rank )

if rank == 0:
    M12_configs = np.zeros_like(pairs_sendbuf)
else:
    M12_configs = None

M12_configs = comm.gather( M12_configs_sendbuf, root=0  )

#print(M12_configs_sendbuf, rank)

# Final handling in rank=0
if rank==0:
    M12_configs = np.concatenate(M12_configs)  

    """
    MM  = np.zeros(n_pairs)
    for i in range(n_pairs):
        MM[i] = Minter_fixed_configs( i )

    assert np.allclose(MM,  M12_configs)
    """
    P1 = softmax( ( -np.array(U1) +np.min(U1) )/T )
    P2 = softmax( ( -np.array(U2) +np.min(U2) )/T )
 
    MinterV = P1.dot( M12_configs.reshape((n_ch1s, n_ch2s)).dot(P2) )
 
    B2V =  -MinterV 

    print('B2V:', B2V)

    f = open( 'B2V_' + seqname1 + '_' + seqname2 + '_T' + str(T) + '_rD' + str(rD) + '_cut' + str(cut) + '.txt', 'w') 
    f.write('B2V:' + str(B2V) )
    f.close() 
