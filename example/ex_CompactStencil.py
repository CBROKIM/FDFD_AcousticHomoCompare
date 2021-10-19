import os,sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

sys.path.append('../src/')
import general
import module,module2D
arr,pi = np.array, np.pi
absmax = lambda x: np.amax(np.abs(x))

def main():
    def Singer2006(kh):
        L,M = module.singer2006(kh,[[0,0],[0,1],[1,1]])
        return L+M
    def Conv2th(kh):
        H = arr([kh*kh-4.0,1.0,0.0])
        return H
    def Jo1996(kh):
        H = module.Jo1996([[0,0],[0,1],[1,1]],kh)
        return H
    C9Schemes = [module.Compact,Singer2006,Jo1996,Conv2th]
    CompactSchemes(C9Schemes, [0.5*pi,0.7*pi])

def CompactSchemes(Schemes,khs,nx=101,nz=101,nPML=20,sx=50.5,sz=50.5):
    for kh in khs:
        fig,ax = plt.subplots(nrows=2,ncols=len(Schemes),sharex=True,sharey=True)
        anal = module2D.Helmholtz2D(*np.meshgrid(np.arange(nx),np.arange(nz)),kh,sx,sz)
        perc = absmax(np.percentile(anal.real,[1,99]))
        fig.suptitle(r'$kh={:05.3f}\pi, 1/G={:05.3f}$'.format(kh/pi,pi/kh*2),fontsize=12)
        for i,scheme in enumerate(Schemes):
            kh = kh*np.ones((nx,nz),dtype=np.float32)
            nx,nz = kh.shape
            S = module2D.Smake(kh,nPML,scheme)
            f = module2D.getf(S,nx,nz,[sx,sz])
            wave = spsolve(S,f).reshape(nx,nz)
            diff = anal-wave 
            ax[0,i].imshow(wave.real,vmin=-perc,vmax=+perc)
            ax[1,i].imshow(diff.real,vmin=-perc,vmax=+perc)
            ax[0,i].set_title(scheme.__name__)
        plt.show()

if __name__=='__main__':
    main()
