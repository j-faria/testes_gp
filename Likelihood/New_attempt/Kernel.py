# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:42:48 2016

@author: camacho
"""
import numpy as np
#from sympy import KroneckerDelta as kd
from time import time   

#####  DADOS INICIAS  #########################################################
#x = 10 * np.sort(np.random.rand(2000))
#yerr = 0.2 * np.ones_like(x)
#y = np.sin(x) + yerr * np.random.randn(len(x))


###############################################################################
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

def likelihood(x, xcalc, y, yerr, kernel, *params):
    #covariance matrix K
    K = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]
            K[i,j] = kernel(x1, x2, *params)
            #print(x1,x2,K[i,j])
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
##### CONCLUSOES
#A minha versão demora cerca de 8 vezes mais 
    
#### Kernels  criadas #########################################################
def ExpSquared(x1, x2, *params): # Squared Exponential Kernel   
    ES_theta, ES_l = params    
    f1 = ES_theta**2
    f2 = ES_l**2
    f3 = (x1 - x2)**2
    return f1 * np.exp(-0.5 * f3 / f2)

def ExpSineSquared(x1, x2, *params): # Periodic Kernel
    ESS_theta, ESS_l, ESS_P = params 
    f1 = ESS_theta**2
    f2 = ESS_l**2
    f3 = (x1-x2)
    return f1*np.exp(-2*(np.sin(np.pi*f3/ESS_P))**2/f2)
       
def RatQuadratic(x1, x2, *params): # Rational Quadratic Kernel
    RQ_theta, RQ_l, RQ_alpha = params
    f1 = RQ_theta**2
    f2 = RQ_l**2
    f3 = (x1-x2)**2
    return f1*(1+(0.5*f3/(RQ_alpha*f2)))**(-RQ_alpha)

def Local_ExpSineSquared(x1, x2, *params):             #Locally Periodic Kernel
    LESS_theta, LESS_l, LESS_P = params                       #identico a fazer
    f1 = LESS_theta**2                               #ExpSineSquared*ExpSquared
    f2 = LESS_l**2
    f3 = (x1-x2)
    f4 = (x1-x2)**2
    return f1*np.exp(-2*(np.sin(np.pi*f3/LESS_P))**2/f2)*np.exp(-0.5*f4/f2)

#i e j = 0 deve calcular mal
#i=0;j=0 #Se não definir nenhum valor inicial dá erro no calculo da likelihood
#def WhiteNoise(WN_theta): # White Noise Kernel
#    return (WN_theta**2)*kd(i,j)#*(x1-x2)
###### A PARTIR DAQUI ACHO QUE NÃO É NECESSARIO MAS DEIXO FICAR NA MESMA ######
## Linear Kernel
#def Linear(x1, x2,L_thetab,L_thetav,L_c):
#    f1 = L_thetab**2
#    f2 = L_thetav**2
#    return f1+f2*(x1-L_c)*(x2-L_c)
#    
##Soma de Periodic com Squared Exponential (ExpSineSquared+ExpSquared)
#def Sum_ExpSineSquared_ExpSquared(x1, x2, ESS_theta, ESS_l, ESS_P, ES_theta, ES_l):
#    return ExpSineSquared(x1,x2,ESS_theta,ESS_l,ESS_P)+ExpSquared(x1,x2,ES_theta,ES_l)
#    
## Squared Exponential Kernel com white noise
#def ExpSquared_WN(x1, x2, ES_theta, ES_l, WN):
#    f1 = ES_theta**2
#    f2 = ES_l**2
#    f3 = (x1-x2)**2
#    return f1*np.exp(-0.5*f3/f2)
#    
## Periodic Kernel com white noise
#def ExpSineSquared_WN(x1, x2, ESS_theta, ESS_l, ESS_P,WN): 
#    f1 = ESS_theta**2
#    f2 = ESS_l**2
#    f3 = (x1-x2)
#    return f1*np.exp(-2*(np.sin(np.pi*f3/ESS_P))**2/f2)


########## CALCULO USANDO O GEOREGE ##########
#import george
#from george.kernels import ExpSquaredKernel
#start = time()
#kernel = ExpSquaredKernel(1.0)
#gp = george.GP(kernel)
#gp.compute(x,yerr)
#print 'Took %f seconds' % (time() - start), ('log_p_george',gp.lnlikelihood(y))


#likelihood(x, x , y , yerr, ExpSquared, 1, 1)
#likelihood(x, x, y, yerr, ExpSineSquared, 19, 2 ,5)