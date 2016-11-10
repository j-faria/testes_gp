# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:48:15 2016

@author: camacho
"""
import Kernel;reload(Kernel);kl = Kernel
import Likelihood as lk
import numpy as np
import matplotlib.pyplot as pl
from time import time   

#def likelihood1(kernel, x, xcalc, y, yerr): #covariance matrix calculations   
#    start = time() #Corrected and faster version      
#    K = np.zeros((len(x),len(x))) #covariance matrix K
#    for i in range(len(x)):
#        x1 = x[i]
#        for j in range(len(xcalc)):                      
#            x2 = xcalc[j]
#            K[i,j] = kernel(x1, x2)
#    K=K+yerr**2*np.identity(len(x)) 
#    tempo=(time() - start)    
#    return tempo
################################################################################
#    
#pontos=[]       
#temposES=[];temposESS=[];temposRQ=[]
#for i in np.arange(50,500,25):
#    pontos.append(i)
#    np.random.seed(100)
#    x = 10 * np.sort(np.random.rand(2*i))
#    yerr = 0.2 * np.ones_like(x)
#    y = np.sin(x) + yerr * np.random.randn(len(x))
#
#    kernel1=kl.ExpSquared(19.0, 2.0)   
#    tempo1=likelihood1(kernel1, x, x, y, yerr)
#    temposES.append(tempo1)
#
#    kernel2=kl.ExpSineSquared(15.0, 2.0, 10.0)
#    tempo2=likelihood1(kernel2, x, x, y, yerr)
#    temposESS.append(tempo2)
#    
#    kernel3=kl.RatQuadratic(1.0,1.5,1.0)
#    tempo3=likelihood1(kernel2, x, x, y, yerr)
#    temposRQ.append(tempo3)
#    
##print temposES
##print temposESS
##print temposRQ   
#
#N=np.log(pontos)
#logES= np.log(temposES)
#logESS= np.log(temposESS)
#logRQ= np.log(temposRQ) 
#
#N2=np.log(pontos)**2
#N3=np.log(pontos)**3

pl.plot(N,logES)
pl.plot(N,logESS)
pl.plot(N,logRQ)
pl.xlabel('Points')
pl.ylabel('Time')
pl.title('Covariance matrix calculations')
pl.legend(['ExpSquared', 'ExpSineSquared', 'RatQuadratic'], loc='upper center')

pl.plot(N2,logES)
pl.plot(N2,logESS)
pl.plot(N2,logRQ)
pl.plot(N3,logES)
pl.plot(N3,logESS)
pl.plot(N3,logRQ)