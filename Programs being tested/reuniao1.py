# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:03:54 2016

@author: camacho
"""

##### LIKELIHOOD
def likelihood1(kernel, x, xcalc, y, yerr): #covariance matrix calculations   
      
    K = np.zeros((len(x),len(x))) #covariance matrix K
    start = time() #Corrected and faster version 
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]
            K[i,j] = kernel(x1, x2)
    K=K+yerr**2*np.identity(len(x)) 
#    start = time() #Corrected and faster version
    log_p_correct = lnlike(K, y)
#    tempo= (time() - start)    
#    print 'likelihood ->', log_p_correct    
    return K

def lnlike(K, r): #log-likelihood calculations
    start = time() #Corrected and faster version
    log_p_correct = lnlike(K, y)
      
    from scipy.linalg import cho_factor, cho_solve
    L1 = cho_factor(K) # tuple (L, lower)
    sol = cho_solve(L1, r) # this is K^-1*(r)
    n = r.size
    logLike = -0.5*np.dot(r, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)       
    tempo= (time() - start)  
    return tempo    
    #return logLikeleft


#pontos=[]       
#temposES=[];temposESS=[];temposRQ=[]
#for i in np.arange(50,500,25):
#    pontos.append(i)
#    np.random.seed(100)
#    x = 10 * np.sort(np.random.rand(2*i))
#    yerr = 0.2 * np.ones_like(x)
#    y = np.sin(x) + yerr * np.random.randn(len(x))
#
#    kernel1=kl.ExpSquared(19.0, 2.0)   
#    tempo1=likelihood1(kernel1, x, x, y, yerr)
#    temposES.append(tempo1)
#
#    kernel2=kl.ExpSineSquared(15.0, 2.0, 10.0)
#    tempo2=likelihood1(kernel2, x, x, y, yerr)
#    temposESS.append(tempo2)
#    
#    kernel3=kl.RatQuadratic(1.0,1.5,1.0)
#    tempo3=likelihood1(kernel2, x, x, y, yerr)
#    temposRQ.append(tempo3)
    
#print temposES
#print temposESS
#print temposRQ   

N=np.log(pontos)
logES= np.log(temposES)
logESS= np.log(temposESS)
logRQ= np.log(temposRQ) 