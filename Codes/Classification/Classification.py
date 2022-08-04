import threading
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
lab=[0, 1]
sub=1
LGBM_accs=0
def cl_LGBM(x):
    global LGBM_accs
    x0=abs(x[0]%30)+0.5
    x1=abs(x[1]%30)+0.5
    x2=x[2]%1+1e-5
    fb,fh=min([x0,x1]),max([x0,x1])
    if fh-fb<1:fh+=1
    NFET,X,Y=processing(X_train,Y_train,fb,fh)
    n=int(4*len(Y)/5)
    X,Y,X_t,Y_t=X[:n],Y[:n],X[n:],Y[n:]
    model = LGBMClassifier(n_estimators=100,max_depth=13,num_leaves=172,
                           learning_rate=x2,
                           reg_alpha=0.0007728039351811225)
    model.fit(X, Y)
    predicted= model.predict(X_t)
    o=accuracy_score(Y_t, predicted)*100
    if o>=LGBM_accs:
        s=[];[s.append(tasks.split(',')[i]) for i in lab]
        model_name=' & '.join(['SUB0%d'%sub,','.join(s),'%.2f'%fb,
                               '%.2f'%fh,'%d'%NFET,'%.2f'%x2])
        threading.Thread(target=Show_results,args=(model_name,predicted,Y_t,)).start()
        LGBM_accs=o
    return 100-o
