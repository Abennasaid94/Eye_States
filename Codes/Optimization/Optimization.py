""" More information for this code is detailed in the following article:
    Title: EEG-based BCI: A novel improvement for EEG signals classification based on real-time preprocessing
    Journal: Computers in Biology and Medicine
    DOI: 10.1016/j.compbiomed.2022.105931
    Publication date: August 2022
"""
import numpy as np
def SCA(Classifier,pos_min,pos_max,dim,pop_size,iter):
    loss_pos=np.zeros(dim)
    loss_score=float("inf")
    pos_min = [pos_min] * dim
    pos_max = [pos_max] * dim
    params = np.zeros((pop_size, dim))
    for i in range(dim):
        params[:, i] = np.random.uniform(0,1, pop_size) * (pos_max[i] - pos_min[i]) + pos_min[i]
    for t in range(iter):
        for i in range(pop_size):
            for j in range(dim):
                params[i,j]=np.clip(params[i,j], pos_min[j], pos_max[j])
            fitness=Classifier(params[i,:])
            if fitness<loss_score:
                loss_score=fitness
                loss_pos=params[i,:].copy()
        for i in range(pop_size):
            for j in range (0,dim):
                r1,r2,r3=np.random.random(),np.random.random(),np.random.random()
                if (r3< (0.5)):
                    params[i,j]= params[i,j]+(2*(1-t*/iter)*np.sin(2*np.pi*r1)*abs(r2*loss_pos[j]-params[i,j]))
                else:
                    params[i,j]= params[i,j]+(2*(1-t*/iter)*np.cos(2*np.pi*r1)*abs(r2*loss_pos[j]-params[i,j]))

def Optimzer():
        Classifier,pos_min,pos_max,dim,popSize,Iter=LGBM,0,1000,3,500,20
        SCA(Classifier,pos_min,pos_max,dim,popSize,Iter)
