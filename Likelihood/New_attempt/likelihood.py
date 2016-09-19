# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:51:12 2016

@author: camacho
"""
import numpy as np
#import scipy  as sp
from Kernel import *
from time import time

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

def likelihood(kernel,x,xcalc,y,yerr):
    #calcular matrix de covariancia K
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(xcalc)):
            x1=x[i]
            x2=xcalc[j]
            K[i,j]=kernel
    K=K+yerr**2*np.identity(len(x))      
    
    #### COMEÇA A MINHA VERSAO
    start = time()
    #para usar cholesky a matriz tem de ser positiva definida
    #L=sp.linalg.lu(K)
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
    
    #assert np.allclose(log_p,log_p_correct)
    #### CONCLUSOES
    #A minha versão demora cerca de 8 vezes mais 