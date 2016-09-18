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

#    def argumentcount(self): #conta as variaveis de x
#        return self.__code__.co_argcount   
#    def addargument(self,i):
#        return args[i]
    
    def __call__(self, *args):
        #arranjar maneira de adicionar args[i]  consoante o valor de i
        return self.f(args[0],args[1]) + self.g(args[2],args[3],args[4])    # funciona
        print(countf)


#@Combinable
def a(x,y):
    return float(x)**2 + x + y

def b(x,y,z):
    return x + float(y)**2 + 1 + z

def c(z):
    return z

d=Combined(a,b)
print(d(1,2,3,4,1))  #funciona
#e=Combined(d,c)
#print(e(1,2,1)) #nao funciona

#def argumentcount(x): #conta as variaveis de x
#    return x.__code__.co_argcount
#run=argumentcount(a)
#
#xx=()
#for i  in range(run):
#    xx=xx+('args[%i]' %i,)
#





###############################################################################
#def argumentcount(x): #conta as variaveis de x
#    return x.__code__.co_argcount

#for i in range(0, 10, 3):
#    pref = "g%02i_" % (i/3)
#print(pref)

###############################################################################
#def callback(fn):
#    def inner(self, *args):
#        return _do_callback(fn.__get__(self, type(self)), self.log, *args)
#    return inner
#
#class Foo(object):
#    def __init__(self):
#        self.log = Log('Foo')
#
#@callback
#def cb1_wrapped(self,x):
#    pass

#def wrap(bound_method):
#    return lambda *args: _do_callback(bound_method, bound_method.__self__.log, args)
#
#dd=callback(d)

######
#def argumentcount(x): #conta as variaveis da função
#    return x.__code__.co_argcount
#print(argumentcount(dd))
######

###############################################################################
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