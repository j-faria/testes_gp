# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:42:48 2016

@author: camacho
"""
import numpy as np
#from sympy import KroneckerDelta as kd
from time import time   

#####  DADOS INICIAS  #########################################################
x = 10 * np.sort(np.random.rand(20))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))


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

def likelihood(kernel,x,xcalc,y,yerr):
    global x1,x2
    #calcular matrix de covariancia K
    K=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        x1=x[i]
        for j in range(len(xcalc)):            
#            x1=x[i]            
            x2=xcalc[j]
            K[i,j]=kernel
            print(x1,x2,K[i,j])

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
    
#Defino como zero inicialmente para nao dar erro no calculo da 
#likelihood por haver variaveis não definidas no inicio.

def ExpSquared(ES_theta, ES_l): # Squared Exponential Kernel   
    global x1, x2    
    f1 = ES_theta**2
    f2 = ES_l**2
    f3 = (x1-x2)**2
    return f1*np.exp(-0.5*f3/f2)
       
def RatQuadratic(RQ_theta, RQ_l, RQ_alpha): # Rational Quadratic Kernel
    global x1, x2 
    f1 = RQ_theta**2
    f2 = RQ_l**2
    f3 = (x1-x2)**2
    return f1*(1+(0.5*f3/(RQ_alpha*f2)))**(-RQ_alpha)

def ExpSineSquared(ESS_theta, ESS_l, ESS_P): # Periodic Kernel
    global x1, x2 
    f1 = ESS_theta**2
    f2 = ESS_l**2
    f3 = (x1-x2)
    return f1*np.exp(-2*(np.sin(np.pi*f3/ESS_P))**2/f2)


def Local_ExpSineSquared(LESS_theta, LESS_l, LESS_P): # Locally Periodic Kernel
    global x1, x2                                             #identico a fazer
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

likelihood(ExpSquared(19,2), x, x, y, yerr)