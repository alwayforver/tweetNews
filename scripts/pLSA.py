import numpy as np
def pLSA_init_km(X,K,Pz_d_km,Pw_z_km):
    Pd = X.sum(axis=1)/X.sum()
    Pd = np.squeeze(np.asarray(Pd))
    Pz_d = Pz_d_km
    Pw_z = Pw_z_km
    return (Pd,Pz_d,Pw_z)
def compute_Pw_d(wordind,docind,Pw_z,Pz_d):
    Pz_dw_ = Pw_z[wordind,:].T * Pz_d[:,docind]
    Pw_d = Pz_dw_.sum(axis=0)  # 1 x nnz
    return Pw_d, Pz_dw_
def EMstep(wordind,docind,indptr,value,Pw_z,Pz_d,Pz_dw_,Pw_d):
    K = len(Pz_d)
    Pz_wd = Pz_dw_/np.tile(Pw_d,(K,1))
    n_wdxPz_wd = np.tile(value,(K,1))*Pz_wd
    nwords = len(Pw_z)
    ndocs = Pz_d.shape[1]
    Pw_z *= 0
    Pz_d *= 0
    for i in range(len(indptr)-1): # indd=i
        stInd = indptr[i]
        enInd = indptr[i+1]
        delta = n_wdxPz_wd[:,stInd:enInd] # add the block of posterior referring a document 
                                          # to corresponding params at once
        Pz_d[:,i] += delta.sum(axis=1)
        indw = wordind[stInd:enInd]
        Pw_z[indw,:] += delta.T
#    for i in range(len(wordind)):
#        indw = wordind[i]
#        indd = docind[i]
#        vec = n_wdxPz_wd[:,i]
#        Pw_z[indw,:] += vec
#        Pz_d[:,indd] += vec
    sumz = Pw_z.sum(axis=0)
    C = np.diag(1/sumz)
    Pw_z = np.dot(Pw_z, C)
#    Pw_z = Pw_z/np.tile(sumz,(nwords,1))
    sumd = Pz_d.sum(axis=0)
    Pz_d = Pz_d/np.tile(sumd,(K,1))
    return Pw_z,Pz_d
def logL(value,Pd_docind,Pw_d):
    Li = (value*np.log(Pw_d * Pd_docind)).sum()
    print Li
    return Li
def pLSA(X,K,Learn,Pz_d_km,Pw_z_km):
    (Min_Likelihood_Change,Max_Iterations) = Learn
    Li=[]
    Pd,Pz_d,Pw_z = pLSA_init_km(X,K,Pz_d_km,Pw_z_km) 
    X = X.tocsr()
    indptr,indices = (X.indptr,X.indices)
    X = X.tocoo()
    docind,wordind,value = (X.row,X.col,X.data)
    Pd_docind = Pd[docind]
    if sum(indices-wordind)!=0:
        print "indices!=wordind"
        print indices
        print wordind
        exit(0)
    Pw_d, Pz_dw_ = compute_Pw_d(wordind,docind,Pw_z,Pz_d)
    for it in range(Max_Iterations):
        print "iteration: "+str(it)
        Pw_z,Pz_d = EMstep(wordind,docind,indptr,value,Pw_z,Pz_d,Pz_dw_,Pw_d)
        Pw_d, Pz_dw_ = compute_Pw_d(wordind,docind,Pw_z,Pz_d)
        Li.append(logL(value,Pd_docind,Pw_d))
        if it > 0:
            dLi = Li[it] - Li[it-1]
            print "dLi = " + str(dLi)
            if dLi < Min_Likelihood_Change:
                break
    print Li[-1]
    return Pw_z,Pz_d,Pd,Li,Learn

