# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:29:15 2016

@author: camacho
"""

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
        return self.f(args[0],args[1]) + self.g(args[2],args[3],args[4])
        #return self.f + self.g

class Mul(Combinable): # nao funciona
    def __init__(self, f, g):
        self.f = f
        self.g = g
    
    def __call__(self, *args):
        return self.f(args[0],args[1]) * self.g(args[2],args[3],args[4])


def a(x,y):
    return float(x)**2 + x + y

def b(x,y,z):
    return x + float(y)**2 + 1 + z

def c(z):
    return z

d=Sum(a,b)
print(d(1,2,3,4,1))  #funciona

e=Mul(a,b) #nao funciona
print(d(1,2,3,4,1))