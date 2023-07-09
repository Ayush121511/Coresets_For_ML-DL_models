import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from matplotlib import image


## Import Dataset
imag = image.imread('peppers.png')
imag.shape
m = imag.shape[0]
n = imag.shape[1]
weight = np.ones((m,n))*1/(m*n)
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
    n = X.shape[1]
    index_set = np.ones((m*n,2))

    for i in range(m*n):
        index_set[i,0] = i
        index_set[i,1] = mu_x[i//n][i%n]

    B_index = np.random.choice( index_set[:,0], p=index_set[:,1],size=1) 
    B = [X[int(B_index[0]//n)][int(B_index[0]%n)]]
    D = np.zeros((m,n))

    for p in range(1,k):
        d_sum = 0
        for i in range(m):
            for j in range(n):
                D[i][j] = d(X[i][j],B)[0]**2*mu_x[i][j]
                d_sum += D[i][j]
        
        for i in range(m*n):
            index_set[i,1] = D[i//n][i%n]/d_sum
        
        B_index = np.random.choice( index_set[:,0], p=index_set[:,1],size=1)
        B.append(X[int(B_index[0]//n)][int(B_index[0]%n)])

    return B


B = d2_sampling(imag,weight,10)


### Coreset Construction
def coresetConstr(X,k,B,f_pr,eps):
    M = (3*k**3*np.log(k) + k**2* np.log(1/f_pr))/ eps**2
    m = X.shape[0]
    n = X.shape[1]
    alpha = 16*(np.log(k) + 2)
    cluster = np.zeros((m,n))
    d_sum=0
    d_f = np.zeros((m,n,2))
    B_i = np.zeros((len(B),2))
    for i in range(m):
        for j in range(n):
            d_f[i][j] = d(X[i][j],B)
            cluster[i][j] = d_f[i][j][1]
            d_sum+=d_f[i][j][0]
            B_i[int(d_f[i][j][1])][1] += 1
            B_i[int(d_f[i][j][1])][0] += d_f[i][j][0]
    c_phi = d_sum / (m*n)
    S = np.zeros((m,n))
    pr = np.zeros((m,n))
    sum_S = 0
    for i in range(m):
        for j in range(n): 
            #Senstivity calculation
            S[i][j] = alpha*d_f[i][j][0]/c_phi + 2*alpha*B_i[int(cluster[i][j])][0]/(B_i[int(cluster[i][j])][1] * c_phi) + 4*m*n/B_i[int(cluster[i][j])][1]
            sum_S += S[i][j]

    for i in range(m):
        for j in range(n):
            #Probability Calculation
            pr[i][j] = S[i][j]/sum_S

    index_set = np.ones((m*n,2))
    for i in range(m*n):
        index_set[i,0] = i
        index_set[i,1] = pr[i//n][i%n]
    
    C_index = np.random.choice( index_set[:,0], p=index_set[:,1],size=int(M)+1)
    C= np.zeros((int(M)+1,3))
    weight = np.zeros((int(M)+1))
    for i in range(int(M+1)):
        C[i] = np.array(X[int(C_index[i]//n)][int(C_index[i]%n)])
        
        #Weight of samples
        weight[i] = 1/(M*pr[int(C_index[i]//n)][int(C_index[i]%n)])

    return C,weight

coreset,weight = coresetConstr(imag,10,B,1,1)


### k-Means Algorithm
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(coreset,sample_weight=weight)
centers = kmeans.cluster_centers_
centers = np.array(centers)
for i in range(m):
    for j in range(n):
        d_f = d(imag[i][j],centers)
        imag[i][j] = centers[int(d_f[1])]


data =  Image.fromarray((imag * 255).astype(np.uint8))
data.save("clustered.png")