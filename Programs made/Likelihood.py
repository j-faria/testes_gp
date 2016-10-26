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
def likelihood_aux(kernel, x, xcalc, y, yerr): #covariance matrix calculations   
    K = np.zeros((len(x),len(x))) #covariance matrix K
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]
            K[i,j] = kernel(x1, x2)
    K=K+yerr**2*np.identity(len(x))      
    log_p_correct = lnlike(K, y) 
    return K

def grad_logp(kernel,x,xcalc,y,yerr,cov_matrix):
    K_grad = np.zeros((len(x),len(x))) 
    for i in range(len(x)):
        x1 = x[i]
        for j in range(len(xcalc)):                      
            x2 = xcalc[j]
            K_grad[i,j] = kernel(x1, x2)
#    K_grad=K_grad
#    print K_grad
    K_inv = np.linalg.inv(cov_matrix)    
    alpha = np.dot(K_inv,y)
    
    #codigo tirado do george
    A = np.outer(alpha, alpha) - K_inv #isto vem do george
    grad_george = 0.5 * np.einsum('ij,ij', K_grad, A) #isto vem do george
    return grad_george
    
def gradient_likelihood(kernel,x,xcalc,y,yerr):
    import inspect
    cov_matrix=likelihood_aux(kernel,x,xcalc,y,yerr)
    if isinstance(kernel,kl.ExpSquared):
        grad1=grad_logp(kernel.dES_dtheta, x, xcalc, y, yerr, cov_matrix)
        grad2=grad_logp(kernel.dES_dl, x, xcalc, y, yerr, cov_matrix)
        print 'gradient ->', grad1, grad2
        return grad1, grad2
    elif isinstance(kernel,kl.ExpSineSquared):
        grad1=grad_logp(kernel.dESS_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dESS_dl,x,xcalc,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dESS_dP,x,xcalc,y,yerr,cov_matrix)
        #print 'gradient ->', grad1, grad2, grad3
        return grad1, grad2, grad3 
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
    elif isinstance(kernel,kl.Sum):
        gradient_sum(kernel,x,xcalc,y,yerr)
    elif isinstance(kernel,kl.Product):
        gradient_mul(kernel,x,xcalc,y,yerr)
                
    else:
        print 'gradient -> We dont need no calculation  \n            We dont need no optimization control'    
    #   Nao apliquei a mesma logica às kernels exponential e matern pois
    #até isto funcionar como deve ser não vale a pena fazer
    #funcionar como deve ser = saber se estou a calcular o gradiente bem
    #e arranjar maneira de isto funcionar com somas e multiplicaçoes de kernels

 
##### LIKELIHOOD GRADIENT FOR SUMS
#EM TESTES - IGNORAR POR ENQUANTO     
def gradient_sum(kernel,x,xcalc,y,yerr):
    print 'Work in progress'
    from numpy import arange
    kernelOriginal=kernel #para nao perder a original com os calculos todos
    a=kernel.__dict__
    grad_result=[]    
    for i in arange(1,len(kernel.__dict__)+1):
        var = "k%i" %i
        k_i = a[var]
        #print var, k_i
        calc = gradient_likelihoodAUX(k_i,x,xcalc,y,yerr,kernelOriginal)
        #print  'calculo', calc
        grad_result.insert(1,calc)
        grad_final  =[]
        for j in range(len(grad_result)):         
           grad_final = grad_final + list(grad_result[j])
    print 'gradient sum ->', grad_final
    return grad_final
#Devolve NoneType -> acontece se faltar return no gradient_likelihood
            
def gradient_likelihoodAUX(kernel,x,xcalc,y,yerr,kernelOriginal):
    import inspect
    cov_matrix=likelihood_aux(kernelOriginal,x,xcalc,y,yerr)
    if isinstance(kernel,kl.ExpSquared):
        grad1=grad_logp(kernel.dES_dtheta, x, xcalc, y, yerr, cov_matrix)
        grad2=grad_logp(kernel.dES_dl, x, xcalc, y, yerr, cov_matrix)
        #print 'gradient ->', grad1, grad2
        return grad1, grad2
    elif isinstance(kernel,kl.ExpSineSquared):
        grad1=grad_logp(kernel.dESS_dtheta,x,xcalc,y,yerr,cov_matrix)
        grad2=grad_logp(kernel.dESS_dl,x,xcalc,y,yerr,cov_matrix)
        grad3=grad_logp(kernel.dESS_dP,x,xcalc,y,yerr,cov_matrix)
        #print 'gradient ->', grad1, grad2, grad3
        return grad1, grad2, grad3 
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
#    elif isinstance(kernel,kl.Sum):
#        gradient_sum(kernel,x,xcalc,y,yerr)
#    elif isinstance(kernel,kl.Product):
#        gradient_mul(kernel,x,xcalc,y,yerr)
                
    else:
        print 'gradient -> We dont need no calculation'    
        print '            We dont need no optimization control'


##### LIKELIHOOD GRADIENT FOR PRODUCTS
#EM TESTES - IGNORAR POR ENQUANTO         
def gradient_mul(kernel,x,xcalc,y,yerr):
    print 'Work in progress'
    from numpy import arange
    a=kernel.__dict__
    grad_result=[]
    for i in arange(1,len(kernel.__dict__)+1):
        var = "k%i" %i
        k_i = a[var]
        print k_i        
        var = "k%i" %i
        k_i = a[var]          