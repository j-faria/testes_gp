# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:45:00 2016

@author: camacho
"""
##### RASCUNHO - ignorar o que for feito aqui #####

import numpy as np
import sympy as sp
from Kernel import *

class Combinable:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        return self.f(args)

    def __add__(self, g):
        return Combined(self.f, g) 


class Combined(Combinable):
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def __call__(self, *args):
        return self.f(args[0]) + self.g(args[1])

#@Combinable
def a(x):
    return x**2 + x

def b(y):
    return float(y**2) + 1

def c(z):
    return z

#def main():
#    d = a + b + c
#    print d(2,1)
#
#if __name__ == "__main__":
#    main()

d=Combined(a,b)
e=Combined(d,c)
print(d(1,2))  #funciona
print(e(1,2,1)) #nao funciona
  
#class combine:                  #somar duas funções com  variaveis iguais
#    def __init__(self, f):
#        self.f = f
#    def __call__(self, x):
#        return self.f(x)
#    def __add__(self, other):
#        return combine(lambda x, y: self(x,y) + other(x,y))

#class combine:                 
#    def __init__(self, f):
#        self.f = f
#    def __call__(self, *pars):
#        return self.f(*pars)
#    def __add__(self, other):
#        return combine(lambda *pars: self + other)  
#        
#@combine
#
#def a(x):
#    return  float(2+x)
#    
#def b(y):
#    return float(y**3)
#    
#c=a+b
#print(c(2)) 
#        
        
        
        
###############################################################################
#import operator
#class operable:                 #permite somar, multiplicar, dividir 
#    def __init__(self, f):
#        self.f = f
#    def __call__(self, x):
#        return self.f(x)
# 
#def op_to_function_op(op):
#    def function_op(self, operand):
#        def f(x):
#            return op(self(x), operand(x))
#        return operable(f)
#    return function_op
# 
#for name, op in [(name, getattr(operator, name)) for name in dir(operator) if "__" in name]:
#    try:
#        op(1,2)
#    except TypeError:
#        pass
#    else:
#        setattr(operable, name, op_to_function_op(op))        
#        
#@operable
#def a(x):
#    return  float(2+x)
#    
#def b(x):
#    return float(x**3)
#    
#c=a+b
#print(c(2))


#x = [-1.5, -1, -0.75, -0.4, -0.25, 0]
#y = [0.55*-3, 0.55*-2, 0.55*-0.6, 0.55*0.4, 0.55*1, 0.55*1.6]
#yerr=0.3 * np.ones_like(x)
#
##pl.plot(x,y,"*")
#
########### definir kernel a usar e parametros #########
#k1 = ExpSquared(x,x,1,1) + ExpSineSquared(x,x,1,2,10)
##k2 = ExpSquared(2,5,x1=x,x2=x)
##kernel=  k1+k2
#
#
##K=np.zeros((len(x),len(x)))
##for i in range(len(x)):
##    for j in range(len(x)):
##        K[i,j]=kernel('theta','l',x1[i],x2[j])
##K=K+yerr**2*np.identity(len(x))      
#
#def f(x,y):
#    return x**2 + y
#def g(z):
#    return z**3
#    
#def h(x,y,z):
#    return f(x,y)*g(z)
#    
#a=np.kron(x,y)