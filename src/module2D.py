import numpy as np
from scipy.sparse import csc_matrix
import  scipy.special as ssp
import matplotlib.pyplot as plt
import module
pi,arr = np.pi,np.array
def main():
    from scipy.sparse.linalg import splu as splu, spsolve
    nx,nz = 50,50
    k=0.5*pi
    kh = k*np.ones((nx,nz),dtype=np.float32)
    S = Smake(kh,10,module.Compact)
    f = getf(S,nx,nz,[25.1,23.9])
    wave = spsolve(S,f).reshape(nx,nz)
    fig,ax = plt.subplots(ncols=2)
    ax[0].imshow(wave.real)
    ax[1].imshow(wave.imag)
    plt.show()

def Helmholtz2D(x,y,k=10.0,x0=0,y0=0):
    r = np.sqrt((x-x0)**2+(y-y0)**2)
    return -0.25j*ssp.hankel2(0,k*r)
def Helmholtz(x,k,s):
    r = np.sqrt((x[0]-s[0])**2+(x[1]-s[1])**2)
    return -0.25j*ssp.hankel2(0,k*r)
def getf(S,n1,n2,sp): 
    f = np.zeros((n1*n2,1),dtype=np.cfloat)
    sx,sz = int(sp[0]),int(sp[1])
    for i1,i2 in [[i,j] for i in [0,1] for j in [0,1]]:
        ix,iz = sx+i1,sz+i2
        ii = ix*n2+iz
        row = S.getrow(ii)
        idx = row.nonzero()[1]
        H = row[0,idx].toarray().ravel()
        k2 = H.sum().real
        xy = np.vstack((idx//n2,idx%n2))
        f[ii]+=np.dot(Helmholtz(xy,np.sqrt(k2),sp),H)
    return f
def Smake(kh, nPML,func=module.Compact,FSB=False):
    n1,n2 = kh.shape
    _kh,index = np.unique(kh, return_index=False,return_inverse=True)
    H=[]
    for k in _kh:
        h = func(k.real)#,False)
        H.append(h)
    H = arr(H,dtype=np.cfloat)[index].T
    H[0] = H[0] - ((kh.imag)**2).ravel()+1j*2*(kh.real*kh.imag).ravel()
    pml = 2.0*1j*np.linspace(1.0/nPML,1.0,nPML)**3
    A1 = np.ones(kh.shape,dtype=np.cfloat)
    A2 = np.ones(kh.shape,dtype=np.cfloat)
    A1[:+nPML,:] -=pml[::-1,np.newaxis] / kh[:nPML,:]   **0.5
    if not FSB:
        A2[:,:+nPML] -=pml[np.newaxis,::-1] / kh[:,:nPML]   **0.5
    A1[-nPML:,:] -=pml[:,np.newaxis] / kh[-nPML:,:]     **0.5
    A2[:,-nPML:] -=pml[np.newaxis,:] / kh[:,-nPML:]     **0.5
    
    Full    = range(n1*n2)
    Hal1 = [n2*i1+i2 for i1 in range(n1-1) for i2 in range(n2)]
    Hal2 = [n2*i1+i2 for i1 in range(n1) for i2 in range(n2-1)]
    Both = [n2*i1+i2 for i1 in range(n1-1) for i2 in range(n2-1)]
    Hal1,Hal2,Both = arr(Hal1),arr(Hal2),arr(Both)
    val,row,col = [],[],[]
    A1,A2 = A1.ravel(),A2.ravel()
    #center
    valtemp = (-2/A2/A2-2/A1/A1+H[0]+4)
    row.extend(Full)
    col.extend(Full)
    val.extend(valtemp[Full])
    #edges
    #   +-n2    , 1-direction
    valtemp = (1/A1/A1+H[1]-1)
    row.extend(Hal1+n2)
    col.extend(Hal1)
    val.extend(valtemp[Hal1+n2])
    row.extend(Hal1)
    col.extend(Hal1+n2)
    val.extend(valtemp[Hal1])
    #   +-1     , 2-direction
    valtemp = (1/A2/A2+H[1]-1)
    row.extend(Hal2+1)
    col.extend(Hal2)
    val.extend(valtemp[Hal2+1])
    row.extend(Hal2)
    col.extend(Hal2+1)
    val.extend(valtemp[Hal2])
    #corners
    valtemp = (H[2])
    row.extend(Both+n2+1)
    col.extend(Both)
    val.extend(valtemp[Both+n2+1])  #1
    row.extend(Both+1)
    col.extend(Both+n2)
    val.extend(valtemp[Both+1])     #3
    row.extend(Both+n2)
    col.extend(Both+1)
    val.extend(valtemp[Both+n2])    #7
    row.extend(Both)
    col.extend(Both+n2+1)
    val.extend(valtemp[Both])       #9
    return csc_matrix((val,(row,col)),shape=(n1*n2,n1*n2))

if __name__=='__main__':
    main()
