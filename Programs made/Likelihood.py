# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:27:49 2016

@author: camacho
"""
import Kernel;reload(Kernel);kl = Kernel
import numpy as np
from time import time   
import inspect as i

##### LIKELIHOOD
def likelihood(kernel, x, xcalc, y, yerr): #covariance matrix calculations   
    K = np.zeros((len(x),len(x))) #covariance matrix K
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]

            K[i,j] = kernel(x1, x2)
    K=K+yerr**2*np.identity(len(x))      
    log_p_correct = lnlike(K, y)
    print 'likelihood ->', log_p_correct    
    return K

def lnlike(K, r): #log-likelihood calculations
    from scipy.linalg import cho_factor, cho_solve
    L1 = cho_factor(K) # tuple (L, lower)
    sol = cho_solve(L1, r) # this is K^-1*(r)
    n = r.size
    logLike = -0.5*np.dot(r, sol) \
              - np.sum(np.log(np.diag(L1[0]))) \
              - n*0.5*np.log(2*np.pi)        
    return logLike

##### LIKELIHOOD GRADIENT
def grad_logp(kernel,x,xcalc,y,yerr,cov_matrix):
    K_grad = np.zeros((len(x),len(x))) 
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]
            K_grad[i,j] = kernel(x1, x2)
    K_inv = np.linalg.inv(cov_matrix)    
    alpha = np.dot(K_inv,y)
    alpha_trans = alpha.T
#formula do gradiente tiradas do Rasmussen&Williams chapter 5, equaçao(5.9)
    grad = 0.5 * np.dot(y.T,np.dot(K_inv,np.dot(K_grad,np.dot(K_inv,y)))) \
            -0.5 * np.einsum('ij,ij',K_inv,K_grad)   
    return grad

def gradient_likelihood(kernel,x,xcalc,y,yerr):
    import inspect
    cov_matrix=likelihood(kernel,x,xcalc,y,yerr)#ele volta a imprimir a likelihood acho que 
                                                #por causa disto mas preciso da matriz de 
                                                #covariancia original
    if isinstance(kernel,kl.ExpSquared):
        grad1=grad_logp(kernel.dES_dtheta, x, xcalc, y, yerr, cov_matrix)
        grad2=grad_logp(kernel.dES_dl, x, xcalc, y, yerr, cov_matrix)
        print 'gradient ->', grad1, grad2
        #return grad1, grad2
    elif isinstance(kernel,kl.ExpSineSquared):
        grad1=grad_logp(kernel.dESS_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dESS_dl,x,xcalc,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dESS_dP,x,xcalc,y,yerr,cov_matrix)
        print 'gradient ->', grad1, grad2, grad3
        #return grad1, grad2, grad3 
    elif isinstance(kernel,kl.RatQuadratic):
        grad1=grad_logp(kernel.dRQ_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dRQ_dalpha,x,xcalc,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dRQ_dl,x,xcalc,y,yerr,cov_matrix)
        print 'gradient ->', grad1, grad2, grad3
        #return grad1, grad2, grad3 
    elif isinstance(kernel,kl.Exponential):
        grad1=grad_logp(kernel.dExp_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dExp_dl,x,xcalc,y,yerr,cov_matrix)
        print 'gradient ->', grad1, grad2
        #return grad1, grad2, grad3 
    elif isinstance(kernel,kl.ExpSineGeorge):
        grad1=grad_logp(kernel.dE_dGamma,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dE_dP,x,xcalc,y,yerr,cov_matrix) 
        print 'gradient ->', grad1, grad2
    else:
        print 'gradient -> We dont need no calculation  \n            We dont need no optimization control'    
    #   Nao apliquei a mesma logica às kernels exponential e matern pois
    #até isto funcionar como deve ser não vale a pena fazer
    #funcionar como deve ser = saber se estou a calcular o gradiente bem
    #e arranjar maneira de isto funcionar com somas e multiplicaçoes de kernels
      
#def gradient_sum(kernel,x,xcalc,y,yerr):
#    #a0=kernel.parSize()
#    a=kernel.__dict__
#    grad_result=[]    
#    for i in range(len(kernel.__dict__)):
##        inv_map = {v: k for k, v in a.iteritems()}
##        print a, inv_map
#        k_i = a.popitem(); k_i = k_i[1]
#        calc = gradient_likelihood(k_i,x,xcalc,y,yerr)
#        grad_result.insert(0,calc)
#        grad_final  =[]
#        for j in range(len(grad_result)):        
#            grad_final = grad_final + list(grad_result[j])
#    print 'gradient sum ->', grad_final
#    return grad_final
##Devolve NoneType, rever!