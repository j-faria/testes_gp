# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:03:54 2016

@author: camacho
"""
import sys; sys.path.append('/home/joao/Work/testes_gp/Trials and attempts/Programs made')
import Kernel;reload(Kernel);kl = Kernel
import Likelihood as lk
import numpy as np
import matplotlib.pyplot as plt
from time import time   
import george as ge
from scipy.linalg import cho_factor, cho_solve



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

      
    L1 = cho_factor(K) # tuple (L, lower)
    sol = cho_solve(L1, r) # this is K^-1*(r)
    n = r.size
    logLike = -0.5*np.dot(r, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)       
    tempo= (time() - start)  
    return tempo    
    #return logLikeleft

class ExpSquared(object):
    def __init__(self, theta, l):
        self.pars = [theta, l]

    def __call__(self, r):
        return self.pars[0]**2 * np.exp(-0.5*r**2/self.pars[1]**2)

class ExpSineSquared(object):
    def __init__(self, theta, l, P):
        self.pars = [theta, l, P]

    def __call__(self, r):
        f1 = self.pars[0]**2
        f2 = self.pars[1]**2
        f4 = self.pars[2]
        return f1*np.exp((-2/f2)*((np.sin(np.pi*np.abs(r)/f4))**2))   
        # return self.pars[0]**2 * np.exp( (-2./self.pars[1]**2) * ((np.sin(np.pi*np.abs(r)/self.pars[2]))**2 ))
        # f1 = self.ESS_theta**2
        # f2 = self.ESS_l**2
        # f3 = np.abs(x1-x2)
        # f4 = self.ESS_P
        # return f1*np.exp((-2/f2)*((np.sin(np.pi*f3/f4))**2))   

# def ES_kernel(r, pars):

# def ESS_kernel(r, pars):


def build_matrix(kernel, x, xcalc, y, yerr): #covariance matrix calculations
    r = x[:, None] - xcalc[None, :]
    K = kernel(r)
    K = K + yerr**2*np.identity(len(x)) 
    return K
    # K = np.zeros((len(x),len(x))) #covariance matrix K
    # for i in xrange(len(x)):
    #     # x1 = x[i]
    #     for j in xrange(len(xcalc)):                      
    #         # x2 = xcalc[j]
    #         K[i,j] = kernel(x[i], xcalc[j])

    # K = K + yerr**2*np.identity(len(x)) 
    # return K 

nrep = 3

pontos=[]       
temposES=[];temposESS=[];temposRQ=[]
georgeES=[];georgeESS=[];georgeRQ=[]

for i in np.arange(100,1000,100):
    print i
    pontos.append(i)
    np.random.seed(100)
    x = 10 * np.sort(np.random.rand(2*i))
    yerr = 0.2 * np.ones_like(x)
    y = np.sin(x) + yerr * np.random.randn(len(x))

    av = []
    for _ in range(nrep):
        start = time()
        kernel1=ExpSquared(19.0, 2.0)   
        build_matrix(kernel1, x, x, y, yerr)
        tempo1= time() - start
        av.append(tempo1)
    temposES.append(sum(av) / float(nrep))


    av = []
    for _ in range(nrep):
        start = time()
        kernel2 = ExpSineSquared(15.0, 2.0, 10.0)
        build_matrix(kernel2, x, x, y, yerr)
        tempo1= time() - start
        av.append(tempo1)
    temposESS.append(sum(av) / float(nrep))
    
    # kernel3=kl.RatQuadratic(1.0,1.5,1.0)
    # tempo3=likelihood1(kernel2, x, x, y, yerr)
    # temposRQ.append(tempo3)

    ###########################################################################
    av = []
    for _ in range(nrep):
        start = time() # Calculation using george 
        kernelg1 = 19**2*ge.kernels.ExpSquaredKernel(2.0**2)
        gp = ge.GP(kernelg1)
        gp.compute(x,yerr)
        gp.lnlikelihood(y)
        tempog1= time() - start
        av.append(tempog1)

    georgeES.append(sum(av) / float(nrep))
    

    av = []
    for _ in range(nrep):
        start = time() # Calculation using george 
        kernelg2 = 15.0**2*ge.kernels.ExpSine2Kernel(2/2.0**2,10.0)
        gp = ge.GP(kernelg2)
        gp.compute(x,yerr)
        # gp.lnlikelihood(y)
        tempog2 = time() - start
        av.append(tempog2)
    georgeESS.append(sum(av) / float(nrep))
    
    # start = time() # Calculation using george 
    # kernelg3 = 1.0**2*ge.kernels.RationalQuadraticKernel(1.5,1.0**2)
    # gp = ge.GP(kernelg3)
    # gp.compute(x,yerr)
    # gp.lnlikelihood(y)
    # tempog3=(time() - start)
    # georgeRQ.append(tempog3)    

    
    
#print temposES
#print temposESS
#print temposRQ   


N = pontos
# N=np.log(pontos)
# logES= np.log(temposES)
# logESS= np.log(temposESS)
# logRQ= np.log(temposRQ) 

# log_geoES= np.log(georgeES)
# log_geoESS= np.log(georgeESS)
# log_geoRQ= np.log(georgeRQ) 

N2=np.log(pontos)**2
N3=np.log(pontos)**3

plt.loglog(N, temposES, 'b-o')
plt.loglog(N, temposESS, 'r-o')
# plt.plot(N,logRQ)
#plt.plot(N2,logES)
#plt.plot(N2,logESS)
#plt.plot(N2,logRQ)
#plt.plot(N3,logES)
#plt.plot(N3,logESS)
#plt.plot(N3,logRQ)
plt.loglog(N,georgeES, 'b--')
plt.loglog(N,georgeESS, 'r--')

# plt.plot(N,log_geoRQ)


# plt.loglog(N, N**2)
# plt.loglog(N, N**3)
plt.xlim(0.9*N[0], 1.1*N[-1])
plt.xlabel('Number of points')
plt.ylabel('Time')
plt.title('Covariance matrix calculations')
plt.legend(['ExpSquared', 'ExpSineSquared', # 'RatQuadratic', \
'george ES','george ESS'],#,'george RQ']
loc='upper left')

plt.show()
