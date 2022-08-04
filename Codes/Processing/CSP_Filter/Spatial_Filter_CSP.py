""" More information for this code is detailed in the following article:
    Title: EEG-based BCI: A novel improvement for EEG signals classification based on real-time preprocessing
    Journal: Computers in Biology and Medicine
    DOI: 10.1016/j.compbiomed.2022.105931
    Publication date: August 2022
"""
import numpy as np
import scipy.linalg as linalg
def CSP(*tasks):
	if len(tasks) < 2:
		return (None,) * len(tasks)
	else:
		iter = range(len(tasks))
		W_i=[]
		for x in iter:
			Sigma_x = Cov_M(tasks[x][0])
			for t in range(1,len(tasks[x])):
				Sigma_x += Cov_M(tasks[x][t])
			Sigma_x = Sigma_x / len(tasks[x])
			count = 0
			negative_Sigma_ = Sigma_ * 0
			for negative_x in [element for element in iter if element != x]:
				for t in range(0,len(tasks[negative_x])):
					negative_Sigma_ += Cov_M(tasks[negative_x][t])
					count += 1
			negative_Sigma_x = negative_Sigma_x / count
			sp_x = Spat_Filter(Sigma_x,negative_Sigma_x)
			W_i += (sp_x,)
			if len(tasks) == 2:
				W_i += (Spat_Filter(negative_Sigma_x,Sigma_x),)
				break
		return W_i

def Cov_M(M):
        return np.dot(M,np.transpose(M))/np.trace(np.dot(M,np.transpose(M)))
def Spat_Filter(Ra,Rb):
	Sigma_ = Sigma_a + Sigma_b
	E,U = linalg.eig(Sigma)
	indices = np.argsort(E)
	indices = indices[::-1]
	E = E[indices]
	U = U[:,indices]
	J = np.dot(np.sqrt(linalg.inv(np.diag(E))),np.transpose(U))
	Sa = np.dot(J,np.dot(Sigma_a,np.transpose(J)))
	Sb = np.dot(J,np.dot(Sigma_b,np.transpose(J)))
	E_1,U_1 = linalg.eig(Sa,Sb)
	indices_1 = np.argsort(E_1)
	indices_1 = indices_1[::-1]
	E_1 = E_1[indices_1]
	U_1 = U_1[:,indices_1]
	sp_a= np.dot(np.transpose(U_1),J)
	return np.real(sp_a)

def csp_train(X,Y,n_fs):
    X,Y=X[:n_fs*int(len(X)/n_fs)],Y[:n_fs*int(len(X)/n_fs)]
    tr=int(X.shape[0]/n_fs)
    n_ch=X.shape[1]
    xx,yy=[],[]
    for i in range(0,len(Y),n_fs):
            yy.append(Y[i])
            xx.append(X[i:i+n_fs].T)
    y=np.array(yy)
    x=np.array(tuple(xx))
    tasks=[]
    for i in range(Y.max()-Y.min()+1):
            a=[]
            for j in range(len(y)):
                    if y[j]==i:
                            a.append(x[j])
            tasks.append(np.array(tuple(a)))
    tasks=tuple(tasks)
    if len(tasks)==2:
        W=np.array(CSP(tasks[0],tasks[1]))
    elif len(tasks)==3:
        W=np.array(CSP(tasks[0],tasks[1],tasks[2]))
    elif len(tasks)==4:
        W=np.array(CSP(tasks[0],tasks[1],tasks[2],tasks[3]))
    elif len(tasks)==5:
        W=np.array(CSP(tasks[0],tasks[1],tasks[2],tasks[3],tasks[4]))
    return W