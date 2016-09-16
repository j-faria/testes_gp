# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:46:50 2016

@author: camacho
"""
import numpy as np
#import matplotlib.pyplot as pl
#pl.close("all") #fecha todas as figuras anteriores
from time import time

import george
from george.kernels import *
from Kernel import *

def lnlike(K, r):
    from scipy.linalg import cho_factor, cho_solve
    L1 = cho_factor(K)  # tuple (L, lower)
    # this is K^-1*(r)
    sol = cho_solve(L1, r)

    n = r.size

    logLike = -0.5*np.dot(r, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)

    return logLike

#### DADOS ####################################################################
#np.random.seed(1000)
x = 10 * np.sort(np.random.rand(2000))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))
#pl.plot(x,y,'.')

#### CALCULO USANDO O GEORGE ##################################################
    #Set up the Gaussian process.
kernel = ExpSquaredKernel(1.0)
gp = george.GP(kernel)
    #Pre-compute the factorization of the matrix.
gp.compute(x,yerr)
    #Compute the log likelihood.
start = time()
log_p_george = gp.lnlikelihood(y)
print 'Took %f seconds' % (time() - start), ('log_p_george',log_p_george)


#### CALCULO  DA LIKELIHOOD ################################################### 
#definir kernel a usar
kernel = ExpSquared 
ES_theta = 1
ES_l = 1
x1=x
x2=x

#calcular matrix de covariancia K
K=np.zeros((len(x),len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        K[i,j]=kernel(x1[i],x2[j],ES_theta,ES_l)
K=K+yerr**2*np.identity(len(x))      


#### COMEÇA A MINHA VERSAO
start = time()
#para usar cholesky a matriz tem de ser positiva definida
L = np.linalg.cholesky(K)
L_inv= np.linalg.inv(L)

y = np.array(y)

#Calculo da log likelihood
n=len(x)
log_p = -0.5*np.dot(np.dot(np.dot(y.T,L.T),L_inv),y) - sum(np.log(np.diag(L))) \
        - n*0.5*np.log(2*np.pi)            

print 'Took %f seconds' % (time() - start), ('log_p',log_p)

#### COMEÇA A VERSAO CORRIGIDA
start = time()
log_p_correct = lnlike(K, y)
print 'Took %f seconds' % (time() - start), ('log_p_correct',log_p_correct)    

assert np.allclose(log_p,log_p_correct, log_p_george)

#### CONCLUSOES
#A minha versão demora cerca de 8 vezes mais 