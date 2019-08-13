#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:10:52 2018

@author: msouza
"""

import scipy
import configparser
from scipy.spatial.distance import pdist,squareform
from multiprocessing import Pool
from functools import partial

config = configparser.ConfigParser()
config.read("parameters.conf")

parms = config['Default']

infile = parms['infile']
Niter = int(parms['Niter'])
Ncpu = int(parms['Ncpu'])
na = int(parms['na']) # ants number
alpha = float(parms['alpha']) # ant trend to follow collective trails
beta = float(parms['beta'])
rho = float(parms['rho']) # pheromone evaporation rate

Q = float(parms['Q']) # pheromone deposit gain
Epsilon = float(parms['Epsilon']) # smallest pheromone deposit
hf_sz = int(parms['hf_sz']) # Hall of fame size

raw_data = scipy.loadtxt(infile,skiprows = 6,usecols = (1,2),comments='E')
dist = squareform(pdist(raw_data))
#dist = scipy.fromfile(infile,sep = " ")
#dist = dist.reshape(312,312)
n = dist.shape[0]
aux = dist.max()
#for i in range(n):
# dist[i,i] = 1000*aux

def Sorteia(k,tau,U,nu = scipy.ones((n,n))):
   p = scipy.zeros(n) 
   
   for j in U:
    p[j] = (tau[k,j]**alpha)*(Q/nu[k,j])**beta
    
   p = scipy.hstack(([0],p/p.sum())).cumsum()
   
   return scipy.where(scipy.rand() > p)[0].argmax()

def GeraSolucoes(i,tau):

   U = list(range(n))
   orig = 0
   l = [orig]
   U.remove(orig)

   while len(U) > 0:
    aux = Sorteia(orig,tau,U,dist)
    l.append(aux)
    U.remove(aux)
    orig = aux
    
   l.append(0) 
   return l 

f = lambda x: scipy.sum([dist[i,j] for i,j in zip(x,x[1:])])
      
AvaliaSolucoes = lambda s: scipy.array([f(x) for x in s])

def AtualizaFeromonios(s,tau,fit):

   for (i,xa) in enumerate(s):
    for j,k in zip(xa,xa[1:]):
     tau[j,k] =  rho*tau[j,k] + Q/float(fit[i]) 
   #idx = scipy.where(tau > Epsilon)
   #tau[idx] = rho*tau[idx]

   return tau

def HF_Updt(hf,x):

 x_fit = int(f(x))
 hf_fit = AvaliaSolucoes(hf).astype(int)
   
 if len(hf) < hf_sz:
   hf.append(x)
   hf.sort(reverse = False)
 else:
   # Hall of fame
   aux = scipy.where(x_fit <= hf_fit)
   if len(aux[0]) != 0:
    i = aux[0].min()
    if x_fit != hf_fit[i]:
     hf.insert(i,x)
     hf.pop()

if __name__ == '__main__':
 
 hf = []
 
 if Ncpu > 1:
  p = Pool(Ncpu)
  
 tau = Epsilon*scipy.ones((n,n))
 
 try:
     for kk in scipy.arange(Niter):
         
      if Ncpu > 1:
       sa = p.map(partial(GeraSolucoes, tau=tau),range(na))
      else:
       sa = [GeraSolucoes(i,tau) for i in range(na)] 
        
      fit = AvaliaSolucoes(sa)
      
      tau = AtualizaFeromonios(sa,tau,fit)
      
      id_max = fit.argmax()
      id_min = fit.argmin()
      print("{:4d} {:5.0f} {:5.0f} {:5.0f}".format(kk,fit[id_max],fit[id_min],fit.mean()))
      HF_Updt(hf,sa[id_min])
 except (KeyboardInterrupt, SystemExit):
      pass
      
 print(hf[0])
 print(AvaliaSolucoes(hf))
      