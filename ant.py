#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:10:52 2018

@author: msouza
"""

import scipy
import pylab

from scipy.spatial.distance import pdist,squareform
from multiprocessing import Pool

raw_data = pylab.loadtxt("a280.tsp",skiprows = 6,usecols = (1,2),comments='E')
data = squareform(pdist(raw_data))

nf = data.shape[0]

for i in range(nf):
 data[i,i] = 1.

#n_hf = 5 # hall of fame (top n_hf best solutions so far)
na = 75 # ants number
alpha = 1.4 # ant trend to follow collective trails
beta = 1.4 # features independence influence
rho = 0.3 # pheromone evaporation rate
Q = 250.  # pheromone deposit gain
# fitness matrix
tau = scipy.rand(nf+1,nf) # pheromone matrix
fit = scipy.rand(na)

def fitness_func(x):
  return pylab.sum([data[i,j] for i,j in zip(x,x[1:])])

def Sorteia(k,tau,nu = None):
   if nu is None:
      nu = pylab.ones(nf)
   p = scipy.array([tau[k,j]**alpha * (1./nu[j])**beta for j in scipy.arange(nf)])
   p = (scipy.hstack(([0],p))/p.sum()).cumsum()
   return scipy.where(scipy.rand() > p)[0].argmax()

def GeraSolucoes(i):
 # Busca o primeiro atributo
   aux = Sorteia(0,tau)
   U = list(range(nf))
   # Busca dos n-1 demais atributos e coloca na lista l

   l = []
   while len(l) < nf:
     if aux in U:
      l.append(aux)
      U.remove(aux)
     #print(data[aux])
     aux = Sorteia(aux,tau[1:,:],data[aux])
   return l

def AvaliaSolucoes(s):
   fit = scipy.array([fitness_func(x) for x in s])
   return fit

def AtualizaFeromonios(s,tau):
   for (i,xa) in enumerate(s):
    k = xa[0]
    tau[0,k] =  tau[0,k] + Q/fit[i]
    tau_s = tau[1:,:]
    for j in xa[1:]:
     tau_s[k,j] =  tau_s[k,j] + Q/fit[i]
     tau_s[j,k] =  tau_s[j,k] + Q/fit[i]
     k = j
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
 for kk in scipy.arange(2000):
  #tt = process_time()
  sa= Pool().map(GeraSolucoes,range(na))
  fit = AvaliaSolucoes(sa)
  tau = AtualizaFeromonios(sa,tau)
  tau = (1. - rho) * tau
  id_max = fit.argmax()
  id_min = fit.argmin()
  print("{:4d} {:5.0f} {:5.0f} {:5.0f} {:5.2f}".format(kk,fit[id_max],fit[id_min],fit.mean(),fit.std()))
 print(sa[fit.argmin()])
  #HF_Updt(hf,[fit[id_max][0],sa[id_max]])
 #print(hf)
