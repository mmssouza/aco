#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:10:52 2018

@author: msouza
"""

import scipy
import pylab
import configparser
from scipy.spatial.distance import pdist,squareform
from multiprocessing import Pool
from functools import partial

raw_data = pylab.loadtxt("a280.tsp",skiprows = 6,usecols = (1,2),comments='E')
dist = squareform(pdist(raw_data))
n = dist.shape[0]
for i in range(n):
 dist[i,i] = 1.

config = configparser.ConfigParser()
config.read("parameters.conf")
parms = config['Default']

#n_hf= 5 # hall of fame (top n_hf best solutions so far)
na = int(parms['na']) # ants number
alpha = float(parms['alpha']) # ant trend to follow collective trails
beta = float(parms['beta']) # ant trend to follow individual shortest path
rho = float(parms['rho']) # pheromone evaporation rate
Q = float(parms['Q']) # pheromone deposit gain
Epsilon = float(parms['Epsilon']) # smallest pheromone deposit


def Sorteia(k,tau,nu = pylab.ones((n,n))):
   p = scipy.array([(tau[k,j]**alpha) * (1./nu[k,j]**beta) for j in range(n)]) 
   p = scipy.hstack(([0],p/p.sum())).cumsum()
   return scipy.where(scipy.rand() > p)[0].argmax()

def GeraSolucoes(i,tau):
   U = list(range(n))
   orig = 0
   l = [orig]
   U.remove(orig)
   
   while len(l) < n:
     aux = Sorteia(orig,tau,dist)
     if aux in U:
      l.append(aux)
      U.remove(aux)
      orig = aux
   return l

def AvaliaSolucoes(s):
   f = lambda x: pylab.sum([dist[i,j] for i,j in zip(x,x[1:])])
   return pylab.array([f(x) for x in s])

def AtualizaFeromonios(s,tau,fit):

   for (i,xa) in enumerate(s):
    for j,k in zip(xa[0:],xa[1:]):
     tau[j,k] =  tau[j,k] + Q/float(fit[i])
     tau[k,j] =  tau[k,j] + Q/float(fit[i])
   idx =  pylab.where(tau>Epsilon)
   tau[idx] = (1.-rho)*tau[idx]
    
   return tau

#def HF_Updt(hf,x):
#  if len(hf) < n_hf:
#   hf.append(x)
#   hf.sort(reverse = True)
#   return


  # Hall of fame
#  aux = scipy.where(x[0] > scipy.array([i[0] for i in hf]))
#  if len(aux[0]) != 0:
#   hf.insert(aux[0].min(),x)
#   hf.pop()

if __name__ == '__main__':
# hf = []
 p = Pool(4)
 tau = Epsilon*pylab.ones((n,n))
 for kk in scipy.arange(500):
  sa = p.map(partial(GeraSolucoes, tau=tau),range(na))
  fit = AvaliaSolucoes(sa)
  tau = AtualizaFeromonios(sa,tau,fit)
  id_max = fit.argmax()
  id_min = fit.argmin()
  print("{:4d} {:5.0f} {:5.0f} {:5.0f} {:5.2f}".format(kk,fit[id_max],fit[id_min],fit.mean(),fit.std()))
 print(sa[fit.argmin()])
  #HF_Updt(hf,[fit[id_max][0],sa[id_max]])
 #print(hf)
