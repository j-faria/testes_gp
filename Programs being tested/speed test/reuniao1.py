# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:03:54 2016

@author: camacho
"""
import Kernel;reload(Kernel);kl = Kernel
import Likelihood as lk
import numpy as np
import matplotlib.pyplot as pl
from time import time   
import george as ge



##### LIKELIHOOD
def likelihood1(kernel, x, xcalc, y, yerr): #covariance matrix calculations   
    start = time() #Corrected and faster version
    K = np.zeros((len(x),len(x))) #covariance matrix K
    start = time() #Corrected and faster version 
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]
            K[i,j] = kernel(x1, x2)
    K=K+yerr**2*np.identity(len(x)) 
#    start = time() #Corrected and faster version
#    log_p_correct = lnlike(K, y)
#    tempo= (time() - start)    
#    print 'likelihood ->', log_p_correct    
#    return K
    r=y

#def lnlike(K, r): #log-likelihood calculations
#    start = time() #Corrected and faster version

      
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


pontos=[]       
temposES=[];temposESS=[];temposRQ=[]
georgeES=[];georgeESS=[];georgeRQ=[]
for i in np.arange(50,500,25):
    pontos.append(i)
    np.random.seed(100)
    x = 10 * np.sort(np.random.rand(2*i))
    yerr = 0.2 * np.ones_like(x)
    y = np.sin(x) + yerr * np.random.randn(len(x))

    kernel1=kl.ExpSquared(19.0, 2.0)   
    tempo1=likelihood1(kernel1, x, x, y, yerr)
    temposES.append(tempo1)

    kernel2=kl.ExpSineSquared(15.0, 2.0, 10.0)
    tempo2=likelihood1(kernel2, x, x, y, yerr)
    temposESS.append(tempo2)
    
    kernel3=kl.RatQuadratic(1.0,1.5,1.0)
    tempo3=likelihood1(kernel2, x, x, y, yerr)
    temposRQ.append(tempo3)
    ###########################################################################
    start = time() # Calculation using george 
    kernelg1 = 19**2*ge.kernels.ExpSquaredKernel(2.0**2)
    gp = ge.GP(kernelg1)
    gp.compute(x,yerr)
    gp.lnlikelihood(y)
    tempog1=(time() - start)
    georgeES.append(tempog1)
    
    start = time() # Calculation using george 
    kernelg2 = 15.0**2*ge.kernels.ExpSine2Kernel(2/2.0**2,10.0)
    gp = ge.GP(kernelg2)
    gp.compute(x,yerr)
    gp.lnlikelihood(y)
    tempog2=(time() - start)
    georgeESS.append(tempog2)
    
    start = time() # Calculation using george 
    kernelg3 = 1.0**2*ge.kernels.RationalQuadraticKernel(1.5,1.0**2)
    gp = ge.GP(kernelg3)
    gp.compute(x,yerr)
    gp.lnlikelihood(y)
    tempog3=(time() - start)
    georgeRQ.append(tempog3)    

    
    
#print temposES
#print temposESS
#print temposRQ   



N=np.log(pontos)
logES= np.log(temposES)
logESS= np.log(temposESS)
logRQ= np.log(temposRQ) 

log_geoES= np.log(georgeES)
log_geoESS= np.log(georgeESS)
log_geoRQ= np.log(georgeRQ) 

N2=np.log(pontos)**2
N3=np.log(pontos)**3

pl.plot(N,logES)
pl.plot(N,logESS)
pl.plot(N,logRQ)
#pl.plot(N2,logES)
#pl.plot(N2,logESS)
#pl.plot(N2,logRQ)
#pl.plot(N3,logES)
#pl.plot(N3,logESS)
#pl.plot(N3,logRQ)
pl.plot(N,log_geoES)
pl.plot(N,log_geoESS)
pl.plot(N,log_geoRQ)
pl.xlabel('Points - log(N)')
pl.ylabel('Time - log(s)')
pl.title('Log marginal likelihood calculations')
pl.legend(['ExpSquared', 'ExpSineSquared', 'RatQuadratic', \
'george ES','george ESS','george RQ'], loc='upper left')

