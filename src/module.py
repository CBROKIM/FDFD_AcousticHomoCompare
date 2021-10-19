import numpy as np
import numpy.linalg as nla
cos,sin = np.cos,np.sin
pi,arr = np.pi,np.array

def near9():
    return arr([[i,j] for i in [0,-1,1] for j in [0,-1,1]])
def near25():
    return arr([[i,j] for i in [0,-1,1,-2,2] for j in [0,-1,1,-2,2]])

def near27():
    return arr([[i,j,k] for i in [0,-1,1] for j in [0,-1,1] for k in [0,-1,1]])
Near9 = near9
Near25 = near25
def Near13():
    Near = [[2,0],[0,2],[-2,0],[0,-2]]
    return np.vstack((Near9(),Near))

def rhomb(M):
    points=[]
    for ix in range(-M,M+1):
        N = M-abs(ix)
        for iz in range(-N,N+1):
            points.append([ix,iz])
    points=np.array(points)
    mid = len(points)//2
    points[[0,mid]]=points[[mid,0]]
    return points

#https://doi.org/10.1190/geo2020-0272.1
def Compact(kh):
    theta = np.linspace(0.0,pi/4., 5)
    x,y = sin(theta),cos(theta)
    A=2*cos(kh*x)+2*cos(kh*y) -4.0 
    B=2*cos(kh*x)*2*cos(kh*y) -4.0
    Mat = np.array([[(A*A).sum(), (A*B).sum()],[(B*A).sum(),(B*B).sum()]])
    rhs = -kh**2*np.array([[A.sum()],[B.sum()]])
    H = nla.solve(Mat,rhs)
    H = np.insert(H,0,kh**2-4*H.sum())
    return H

#https://doi.org/10.1137/120902689
def Turkel2013(kh):
    A = [-64/15+14*kh**2/15-kh**4/20,\
         7/15-kh**2/90,\
         1/10+kh**2/90,\
         1/30]
    return arr(A)

#https://doi.org/10.1016/j.jcp.2016.03.023
def STOLK2016(kh):
    Mat = arr([
    [0.0000,0.635413,-0.000228,0.210638, 0.016303,0.172254,-0.014072,0.710633,-0.006278,0.245303,0.019576],\
    [0.0500,0.635102,-0.015578,0.210152,-0.023424,0.171912,-0.005802,0.709821,-0.047764,0.245148,0.021398],\
    [0.1000,0.634166,-0.034804,0.208167,-0.043396,0.171146,-0.012462,0.707374,-0.070981,0.244762,0.007493],\
    [0.1500,0.632093,-0.054496,0.205348,-0.065935,0.170031,-0.022145,0.703359,-0.088202,0.245160,0.009937],\
    [0.2000,0.628341,-0.103457,0.201605,-0.069385,0.169740, 0.001893,0.698813,-0.092327,0.245687,0.012201],\
    [0.2500,0.622526,-0.133896,0.197423,-0.098212,0.169475,-0.002559,0.694726,-0.066617,0.246454,0.016791],\
    [0.3000,0.614611,-0.183988,0.192414,-0.115398,0.168690,-0.005589,0.692615,-0.011177,0.247743,0.029213],\
    [0.3500,0.603680,-0.255991,0.186819,-0.120930,0.167581,-0.015564,0.694109, 0.077605,0.250098,0.059733],\
    [0.4000,0.588498,-0.356326,0.180737,-0.132266,0.166640,-0.001852,0.700902, 0.199685,0.254352,0.106049]])
    Ginv = kh/pi/2
    Gidx = min(int(Ginv*20),8)
    dx = Ginv*20-Gidx
    if Gidx<8:
        f = Mat[Gidx:Gidx+2,1::2].T
        df= Mat[Gidx:Gidx+2,2::2].T
        a1,a2,a3,a4,a5 = Hermite(f,df/20,dx)
    else:
        a1,a2,a3,a4,a5 = Mat[-1,1::2]+Mat[-1,2::2]*dx/20
    kh2 = kh*kh
    H = arr([6*a4-kh2*a1,\
            -a4+a5-kh2*a2/6,\
            -1/2*a5+1/2*(1-a4-a5)-kh2/12*a3,\
            -3/4*(1-a4-a5)-kh2/8*(1-a1-a2-a3)])
    return H*kh**2/(H[0]+6*H[1]+12*H[2]+8*H[3])
def Hermite(F,dF,x):
    if F.ndim ==1:
        F,dF = [F],[dF]
    a = []
    for f,df in zip(F,dF):
        a.append(arr([[x**3,x**2,x,1]])@arr([[2,-2,1,1],[-3,3,-2,-1],[0,0,1,0],[1,0,0,0]])@(arr([[f[0],f[1],df[0],df[1]]]).T))
    return np.squeeze(arr(a))

def OPERTO2007(kh):
    L,M = Operto(kh)
    return L+M

#https://doi.org/10.1190/1.2759835
def Operto(kh):
    wm = [0.4964958,0.4510125,0.052487,0.45523*1e-5]
    w  = [1.8395265*1e-5,0.890077,0.1099046]
    M = np.array([wm[0],wm[1]/6,wm[2]/12,wm[3]/8])
    L = np.array([-6*w[0]-12/3*w[1]-12/4*w[2],\
                  w[0]+1/3*w[1]+2/4*w[2],\
                  1/2/3*w[1]-1/4*w[2],\
                  3/2/4*w[2]])
    return L,M*kh*kh

#https://doi.org/10.1016/j.jcp.2005.06.011
def tsukerman2006(kh):
    def e(g):
        return np.exp(2**g *1j*kh)
    H0 = (e(1/2)+1)*(e(1/2)*e(1)+2*e(1/2)*e(0)-4*e(-1/2)*e(1)+e(1/2)-4*e(-1/2)+e(1)+2*e(0)+1)
    H1 = -(e(3/2)*e(0)-2*e(1/2)*e(1)+2*e(1/2)*e(0)-2*e(1/2)+e(0))
    H2 = e(-1/2)*(2*e(1/2)*e(0)-e(-1/2)*e(1)-2*e(-1/2)*e(0)-e(-1/2)+2*e(0))
    H = np.array([H0,H1,H2])/(e(0)-1)**2 /(e(-1/2)-1)**4
    H = kh*kh*H/(H[0]+4*H[1]+4*H[2])
    L = np.array([-4,1,1])
    return L,H-L

#https://doi.org/10.1190/1.1443979
def Jo1996(points,kh):
    kh=np.array(kh)
    a=0.5713
    c=0.6274
    d=0.09381
    K= np.array([-4*a -4*(1-a)/2,a,(1-a)/2])
    M= np.array([c,d,(1-c-4*d)/4])*kh*kh
    points = np.absolute(points)
    i = points.T[0]+points.T[1]
    return (M+K)[i]

def p17():
    p = []
    for ix in [0,1,-1]:
        for iz in [0,1,-1]:
            p.append((ix,iz))
    for i in [-2,2]:
        p.append((i,0))
        p.append((0,i))
        p.append((i,i))
        p.append((i,-i))
    return np.array(p)

#https://doi.org/10.1190/geo2014-0124.1
def tang2015(kh,p):
    alpha=[0.984854,0.848986,0.500001,0.307729]
    beta =[0.984854,0.848986,0.500001,0.307729]
    b    = 0.517610
    c    = 0.104113
    d    = 0.026328
    e    =-0.010363
    f    = 0.000580
    A = 2*alpha[0]-1
    M = [[b,c,e],\
         [c,d,0],\
         [e,0,f]]
    M=np.array(M)
    a = alpha
    a1,a2,a3,a4 = a[0],a[1],a[2],a[3]
    P0 = np.array([[2*a3,a4,0.5-a4-a3/2],\
                  [a4,0,0],\
                  [0.5-a4-a3/2,0,0]])
    P1 = np.array([[0,a2,0],\
                   [a2,1-a2,0],\
                   [0,0,0]])
    P2 = np.array([[0,0,a1],\
                   [0,0,0],\
                   [a1,0,1-a1]])
    K = -5/2*P0+4/3*P1-1/12*P2
    p = np.absolute(p).astype(int)
    ix,iz = p.T[0],p.T[1]
    return K[ix,iz],M[ix,iz]*kh*kh

def Li2019M2(points,kh):
    L,M = Li2019(points,kh,2)
    return L+M
def Li2019M3(points,kh):
    L,M = Li2019(points,kh,3)
    return L+M

#https://doi.org/10.1016/j.jappgeo.2019.05.002
def Li2019(points,kh,order=1):
    points = np.absolute(points).astype(int)
    if order==1:
        L = [[-3.9992,0.9998],
             [ 0.9998,0.0000]]
        M = [[ 0.7356,0.0661],
             [ 0.0661,0.0000]]
    if order==2:
        L = [[-1.9952,0.1375,0.0702],
             [ 0.1375,0.2911,0.0000],
             [ 0.0702,0.0000,0.0000]]
        M = [[ 0.4576,0.1195,-0.0010],
             [ 0.1195,0.0171,0.0000],
             [-0.0010,0.0000,0.0000]]
    if order==3:
        L = [[-0.7176,-0.1011,0.0741,0.0002],
             [-0.1011, 0.0766,0.0648,0.0000],
             [ 0.0741, 0.0648,0.0000,0.0000],
             [ 0.0002, 0.0000,0.0000,0.0000]]
        M = [[0.2504,0.1176,0.0114,0.0004],
             [0.1176,0.0538,0.0021,0.0000],
             [0.0114,0.0021,0.0000,0.0000],
             [0.0004,0.0000,0.0000,0.0000]]
    L,M = np.array(L),np.array(M)*kh**2
    ix,iz = points.T[0],points.T[1]
    return L[ix,iz],M[ix,iz]

#https://doi.org/10.1190/1.1444323
def ShinSohn(points,kh):
    points = np.absolute(points).astype(int)
    a = [0.0949098,0.280677,0.247253,0.0297411,0.173708,0.173708]
    b = [0.363276,0.108598,0.0041487,0.0424801,0.000206312,0.00187765,0.00188342]
    M = [[b[0],b[1],b[2]],
         [b[1],b[3],b[5]],
         [b[2],b[6],b[4]]]
    L = [[-4.0*(a[0]/1.0+a[1]/4.0+a[2]/2.0+a[4]/5.0+a[5]/5.0+a[3]/8.0),
                   a[0]/1.0,a[1]/4.0],
         [a[0]/1.0,a[2]/2.0,a[4]/5.0],
         [a[1]/4.0,a[5]/5.0,a[3]/8.0]]
    a,b = np.array(a),np.array(b)
    L,M = np.array(L),np.array(M)
    ix,iz = points.T[0],points.T[1]
    return L[ix,iz]+M[ix,iz]*kh**2

#https://doi.org/10.1142/S0218396X06003050
def singer2006(kh,p=near9()):
    r = 11/12
    L = np.array([-10.0/3.0, 2.0/3.0,1.0/6.0])
    M = np.array([67./90.,      2./45.,         7./360.])*kh**2+\
        np.array([(r-3.)/180.,  (3.-2.*r)/720., r/720.])*kh**4
    idx = [int(i) for i in np.sum(np.abs(p),axis=1)]
    return L[idx],M[idx]

def gauss(time, wlength, delay=None):
        if delay == None:
            delay = 3*wlength
        t = time - delay
        return -((2*np.sqrt(np.e)/(wlength)) * t*np.exp(-2*(t**2)/(wlength**2)))

def source(dt=0.01,Nt=100,peak_f=50.0):
    t=np.arange(Nt)*dt
    f=np.linspace(0.,0.5/dt,int(Nt/2))
    wt=gauss(t,1./peak_f,4./peak_f)
    wf=np.fft.fft(wt)
    return wt,wf,t,f

if __name__=='__main__':
    main()
