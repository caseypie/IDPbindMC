import sys
import numpy as np
import seq_list as sl


# Charged polymer model based on Gaussian chain
# l = bare Kuhn length / unit length
class ChargedPolymer:
   def __init__(self, sig, l=1):
       self.sig = sig
       self.N   = sig.shape[0]
       self.l   = l
       self.XYZ   = np.zeros( (self.N, 3) )
 
       # Generate a Gaussian chain
       dXYZ = np.random.normal(scale=self.l/np.sqrt(3), size=(self.N-1,3))
       self.XYZ[1:,0] = np.cumsum(dXYZ[:,0])
       self.XYZ[1:,1] = np.cumsum(dXYZ[:,1])
       self.XYZ[1:,2] = np.cumsum(dXYZ[:,2])
 
   
# Interchain electric energy 
# rD        = Debye screening length
# R1diff = (dX1, dY1, dZ1) between the first monomers in the two polymers
def Uel_inter( Poly1, Poly2, R1diff=(0,0,0), rD=3 ):  
    N1, N2 = Poly1.N, Poly2.N
    XYZ1, XYZ2 = Poly1.XYZ, Poly2.XYZ
    
    Xdiff = np.tile(XYZ1[:,0],(N2,1)).T - np.tile(XYZ2[:,0] + R1diff[0],(N1,1))
    Ydiff = np.tile(XYZ1[:,1],(N2,1)).T - np.tile(XYZ2[:,1] + R1diff[1],(N1,1))
    Zdiff = np.tile(XYZ1[:,2],(N2,1)).T - np.tile(XYZ2[:,2] + R1diff[2],(N1,1))
    Rdiff = np.sqrt(Xdiff**2 + Ydiff**2 + Zdiff**2)

    SIGMA = np.tile(Poly1.sig,(N2,1)).T * np.tile(Poly2.sig,(N1,1))
   
    nRsmall = np.where(Rdiff < 1)[0].shape[0] 
    U = np.sum( SIGMA * np.exp(-Rdiff/rD)/Rdiff )*(nRsmall==0) + 1e20*( nRsmall>0 )

    return U

# Intrachain electric energy 
# rD        = Debye screening length
def Uel_intra( Poly, rD=3 ): 
    N, XYZ, sig = Poly.N, Poly.XYZ, Poly.sig
    Xa = np.tile(XYZ[:,0],(N,1))
    Ya = np.tile(XYZ[:,1],(N,1))
    Za = np.tile(XYZ[:,2],(N,1))
    Sa = np.tile(sig,(N,1))

    Xdiff = Xa.T - Xa
    Ydiff = Ya.T - Ya
    Zdiff = Za.T - Za
    Rdiff = np.sqrt(Xdiff**2 + Ydiff**2 + Zdiff**2)
    SIGMA = Sa.T * Sa

    Id = np.identity(N)
    U = 0.5*np.sum( SIGMA * np.exp(-Rdiff/rD)/ ( Rdiff + Id) * (np.ones((N,N)) - Id) )

    return U
    