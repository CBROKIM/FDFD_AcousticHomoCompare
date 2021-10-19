import numpy as np
import numpy.linalg as nla
cos,sin = np.cos,np.sin
pi=np.pi

def HelmOpt(points,k):
    DerMat = formDerMat(points,2)
    Null= NullMat(DerMat[[0]])
    H = np.zeros((len(points),1))
    H[0]=k*k
    U=[]
    x,y = points.T[0]-points[0,0], points.T[1]-points[0,1]
    for theta in np.linspace(0,np.pi*2,len(points)+21,endpoint=False):
        u = np.exp(-1j*k*(np.cos(theta)*x+np.sin(theta)*y))
        U.append(u)
    U = np.array(U)
    UNULL = U@Null
    x,_,_,_ = nla.lstsq(UNULL,-U@H,rcond=None)
    H = ((H+Null@x)).real
    return H.ravel()

def HelmLM(points,k):
    DerMat = formDerMat(points,2)
    Null= NullMat(DerMat[[0]])
    Null= NullMat(DerMat[[0,1,2]])
    H = np.zeros((len(points),1))
    H[0]=k*k
    U=[]
    x,y = points.T[0]-points[0,0], points.T[1]-points[0,1]
    for theta in np.linspace(0,np.pi*2,41,endpoint=False):
        u = np.exp(-1j*k*(np.cos(theta)*x+np.sin(theta)*y))
        U.append(u)
    U = np.array(U)
    UNULL = U@Null
    x,_,_,_ = nla.lstsq(UNULL,-U@H,rcond=None)
    H = ((H+Null@x)).real
    L = FDM(2,points,2,[[2,0],[0,2]]).ravel()
    M = np.zeros_like(L)
    M[0] = k*k
    Null = NullMat(DerMat[[0,1,2,3,5]])
    x = nla.lstsq(Null.T@Null, Null.T@((H.ravel()-L-M)[:,None]),rcond=None)[0]
    L = L+(Null@x).ravel().real
    M=H.ravel()-L
    return L,M

def NullMat(Mat):
    N = len(Mat[0])
    RREFMat,dep,idx = RREF(Mat)
    ind = np.delete(range(N),dep)
    Null = -RREFMat[:,ind]
    for i,vec in enumerate(np.eye(len(ind))):
        Null = np.insert(Null,ind[i],vec,axis=0)
    return Null
   
def formDerMat(points,Trunc=3):
    Dim = len(points[0])
    delta = np.array(points-points[0])
    DerMat=[]
    alphas = findAlpha(Dim,Trunc)
    for alpha in alphas:
        row = []
        for x in delta:
            row.append(np.prod(x**alpha))
        DerMat.append(np.array(row)/vfact(alpha))
    return np.array(DerMat)

def sub_Alpha(Res,Dim,arr,result):
    if Dim == 1:
        arr = arr + [Res]
        result += [arr]
    else:
        for n in range(Res,-1,-1):
            sub_Alpha(Res-n, Dim-1, arr+[n],result)

def findAlpha(Dim, Order):
    result = [[0 for i in range(Dim)] ]
    for order in range(1,Order+1):
        for n in range(order,-1,-1):
            sub_Alpha(order-n,Dim-1,[n],result)
    return np.array(result)

def RREF(M,augment=False, tol = 1e-10):
    M = np.array(M).astype(float)
    row,col = M.shape
    lead = 0
    tt = []
    idx = np.arange(row)
    if augment==True:
        col-=1
    for c in range(col):
        v = 1.0
        if lead == row:
            break
        MaxIdx = np.argmax(abs(M[lead:,c]))
        MaxVal = M[lead + MaxIdx,c]
        if abs(MaxVal)<tol:
            M[lead:,c] = 0.0
        else:
            idx[lead],idx[lead+MaxIdx] = idx[lead+MaxIdx].copy(), idx[lead].copy()
            M[lead,c:], M[lead+MaxIdx,c:] = M[lead+MaxIdx,c:].copy(), M[lead,c:].copy()
            for r1 in range(lead):
                M[r1,c:] -= M[r1,c] / M[lead,c] * M[lead,c:]
            for r2 in range(lead+1,row):
                M[r2,c:] -= M[r2,c] / M[lead,c] * M[lead,c:]
            M[lead,c:] /= M[lead,c]
            tt.extend([c])
            lead = lead + 1
    M[abs(M)<tol] = 0.0
    return M, np.array(tt).astype(int), idx[:len(tt)]

def vfact(vec):
    result = 1
    for n in vec:
        for i in range(n,1,-1):
            result*=i
    return result

def FDM(Dim,grid,Trunc=2, Target = np.array([[2,0],[0,2]])):
    Target = np.reshape(Target,(-1,Dim))
    delta=np.array(grid)
    KMat=[]
    alphas = findAlpha(Dim,Trunc)
    for alpha in alphas:
        row = []
        for x in delta:
            row.append(np.prod(x**alpha))
        KMat.append(np.array(row)/vfact(alpha))
    RHS = np.zeros(len(KMat),dtype=bool)
    for a in Target:
        RHS = np.bitwise_or(RHS,np.bitwise_and.reduce(alphas==a,1))
    RHS = np.reshape(RHS.astype(float),(-1,1))
    x,_,_,_ = nla.lstsq(KMat,RHS,rcond=None)
    return np.array(x).ravel()
