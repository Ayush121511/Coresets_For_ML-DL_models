import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from matplotlib import image
import pandas as pd

## Import Dataset
data= pd.read_csv("dataset.csv",sep=";")
data
X = data.to_numpy()
deleted=0
arr_aug = X
for i in range(X.shape[0]):
    if(np.any(X[i]!=X[i])):
        arr_aug = np.delete(arr_aug,i-deleted,axis=0)
        deleted+=1
X=arr_aug
unique = list(np.unique(X[:,7]))

for i in range(X[:,7].shape[0]):
    X[i,7] = unique.index(X[i,7])
def d(x,B):
    min_dist = 1e+5
    B_close = -1
    for i in range(len(B)):
        dist = dist = np.linalg.norm(x-B[i])
        if(dist<min_dist):
            min_dist = dist
            B_close = i
    return dist,B_close

### D2_Sampling
def d2_sampling(X,mu_x,k):

    m = X.shape[0]
    index_set = np.ones((m,2))

    for i in range(m):
        index_set[i,0] = i
        index_set[i,1] = mu_x[i]

    B_index = np.random.choice( index_set[:,0], p=index_set[:,1],size=1) 

    B = [X[int(B_index)]]
    D = np.zeros((m,))

    for p in range(1,k):
        d_sum = 0
        for i in range(m):
            D[i] = d(X[i],B)[0]**2*mu_x[i]
            d_sum += D[i]
        
        for i in range(m):
            index_set[i,1] = D[i]/d_sum
        
        B_index = np.random.choice( index_set[:,0], p=index_set[:,1],size=1)
        B.append(X[int(B_index)])

    return B


m = X.shape[0]
weight = np.ones((m))*1/(m)
B = d2_sampling(X,weight,10)

### Coreset Construction
def coresetConstr(X,k,B,f_pr,eps):
    M = (3*k**3*np.log(k) + k**2* np.log(1/f_pr))/ eps**2
    m = X.shape[0]
    alpha = 16*(np.log(k) + 2)
    cluster = np.zeros((m))
    d_sum=0
    d_f = np.zeros((m,2))
    B_i = np.zeros((len(B),2))
    for i in range(m):
        d_f[i] = d(X[i],B)
        cluster[i] = d_f[i][1]
        d_sum+=d_f[i][0]
        B_i[int(d_f[i][1])][1] += 1
        B_i[int(d_f[i][1])][0] += d_f[i][0]
    c_phi = d_sum / (m)
    S = np.zeros((m,))
    pr = np.zeros((m,))
    sum_S = 0
    for i in range(m):
        #Senstivity calculation
        S[i] = alpha*d_f[i][0]/c_phi + 2*alpha*B_i[int(cluster[i])][0]/(B_i[int(cluster[i])][1] * c_phi) + 4*m/B_i[int(cluster[i])][1]
        sum_S += S[i]

    for i in range(m):
        #Probability Calculation
        pr[i] = S[i]/sum_S

    index_set = np.ones((m,2))
    for i in range(m):
        index_set[i,0] = i
        index_set[i,1] = pr[i]
    
    C_index = np.random.choice( index_set[:,0], p=index_set[:,1],size=int(M)+1)
    C= np.zeros((int(M)+1,X.shape[1]))
    weight = np.zeros((int(M)+1))

    for i in range(int(M+1)):
        C[i] = np.array(X[int(C_index[i])])
        
        #Weight of samples
        weight[i] = 1/(M*pr[int(int(C_index[i]))])

    return C,weight
coreset,weight = coresetConstr(X,10,B,1,1)

### k-Means Algorithm
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(coreset,sample_weight=weight)
centers = kmeans.cluster_centers_
centers = np.array(centers)
center_index = list(np.unique(centers))
labels = np.zeros((m,X.shape[1]+1))
for i in range(m):
    d_f = d(X[i],centers)
    labels[i,:-1] = centers[int(d_f[1])]
    labels[i][X.shape[1]] = int(d_f[1])


df = pd.DataFrame(labels, columns = ['year',	'month',	'day',	'order',	'country',	'session ID',	'page 1 (main category)',	'page 2 (clothing model)',	'colour',	'location',	'model photography',	'price',	'price 2'	,'page', 'label ID'])
df.to_csv('labels.csv')