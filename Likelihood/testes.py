# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:45:00 2016

@author: camacho
"""
##### RASCUNHO - ignorar o que for feito aqui #####

import numpy as np
import sympy as sp
from Kernel import *


x = [-1.5, -1, -0.75, -0.4, -0.25, 0]
y = [0.55*-3, 0.55*-2, 0.55*-0.6, 0.55*0.4, 0.55*1, 0.55*1.6]
yerr=0.3 * np.ones_like(x)

#pl.plot(x,y,"*")

########## definir kernel a usar e parametros #########
k1 = ExpSquared(1,1,4,3) 
#k2 = ExpSquared(2,5,x1=x,x2=x)
#kernel=  k1+k2


#K=np.zeros((len(x),len(x)))
#for i in range(len(x)):
#    for j in range(len(x)):
#        K[i,j]=kernel('theta','l',x1[i],x2[j])
#K=K+yerr**2*np.identity(len(x))      

def f(x,y):
    return x**2 + y
def g(z):
    return z**3
    
def h(x,y,z):
    return f(x,y)+g(z)
    
a=np.kron(x,y)