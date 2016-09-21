# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:40:14 2016

@author: camacho
"""
import numpy as np
#####  DADOS INICIAS  #########################################################
x = 10 * np.sort(np.random.rand(20))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))
###############################################################################

class Combinable:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        return self.f(args)

    def __add__(self, g):
        return Sum(self.f,g)    
        
    def __mul__(self, g):
        return Mul(self.f, g)

class Sum(Combinable): # funciona
    def __init__(self, f, g):
        self.f = f
        self.g = g
    
    def __call__(self, *args):
        #arranjar maneira de adicionar args[i]  consoante o valor de i
        return self.f + self.g
        #return self.f + self.g

class Mul(Combinable): # nao funciona
    def __init__(self, f, g):
        self.f = f
        self.g = g
    
    def __call__(self, *args):
        return self.f * self.g


##### Kernels  criadas ########################################################
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

##### LIKELIHOOD
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

def likelihood( x, xcalc, y, yerr, kernel, *params):
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