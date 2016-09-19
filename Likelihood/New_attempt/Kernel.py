# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:42:48 2016

@author: camacho
"""
import numpy as np
from sympy import KroneckerDelta as kd

x1=0;x2=0 #Defino como zero inicialmente para nao dar erro no calculo da 
#likelihood por haver variaveis não definidas no inicio.

def ExpSquared(ES_theta, ES_l): # Squared Exponential Kernel
    f1 = ES_theta**2
    f2 = ES_l**2
    f3 = (x1-x2)**2
    return f1*np.exp(-0.5*f3/f2)
       
def RatQuadratic(RQ_theta, RQ_l, RQ_alpha): # Rational Quadratic Kernel 
    f1 = RQ_theta**2
    f2 = RQ_l**2
    f3 = (x1-x2)**2
    return f1*(1+(0.5*f3/(RQ_alpha*f2)))**(-RQ_alpha)

def ExpSineSquared(ESS_theta, ESS_l, ESS_P): # Periodic Kernel 
    f1 = ESS_theta**2
    f2 = ESS_l**2
    f3 = (x1-x2)
    return f1*np.exp(-2*(np.sin(np.pi*f3/ESS_P))**2/f2)


def Local_ExpSineSquared(LESS_theta, LESS_l, LESS_P): # Locally Periodic Kernel
#identico a fazer ExpSineSquared*ExpSquared
    f1 = LESS_theta**2
    f2 = LESS_l**2
    f3 = (x1-x2)
    f4 = (x1-x2)**2
    return f1*np.exp(-2*(np.sin(np.pi*f3/LESS_P))**2/f2)*np.exp(-0.5*f4/f2)

i=0;j=0 #Se não definir nenhum valor inicial dá erro no calculo da likelihood
def WhiteNoise(WN_theta): # White Noise Kernel
    return (WN_theta**2)*kd(i,j)#*(x1-x2)

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
