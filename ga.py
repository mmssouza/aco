#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:03:24 2019

@author: msouza
"""

import pylab
from scipy.spatial.distance import pdist,squareform

infile = "/home/msouza/prj/aco/a280.tsp"
raw_data = pylab.loadtxt(infile,skiprows = 6,usecols = (1,2),comments='E')
dist = squareform(pdist(raw_data))
ni = dist.shape[0]

Niter = 3500
nn = 150 # number of offsprings


costfunc = lambda x: pylab.sum([dist[i,j] for i,j in zip(x,x[1:])])
 

def op_mutation2(r):
    
  if pylab.rand() < P_mutation2:
   i,j = None,None  
   
   while True: 
    i,j = pylab.randint(low = 1,high = ni-1,size = 2) 
    if i != j:
     break   

   if i > j:
    hi = i
    lo = j
   else:
    lo = i
    hi = j 
    
   aux1,aux2 =  list(r)[:lo],list(r)[hi:]
   aux1.reverse()
   aux2.reverse()
   
   return aux1 + list(r)[lo:hi] + aux2 

  else:
   return r

def op_mutation(r):

  r = op_mutation2(r)
  
  for i,x in enumerate(r):
   if pylab.rand() < P_mutation:
     if i > 0:
      aux = r[i-1]
      r[i-1] = x
      r[i] = aux
     else:
      aux = r[-1]
      r[-1] = x
      r[i] = aux
     
  return r
 
def op_crossover(r1,r2):
  
  while True: 
   i,j = pylab.randint(low = 1,high = ni-1,size = 2) 
   if i != j:
    break   

  if i > j:
   hi = i
   lo = j
  else:
   lo = i
   hi = j   
 
  of2 = list(r2)[lo:hi]
  of1 = list(r1)[lo:hi]

  aux2 = list(r2)[hi:]+list(r2)[:hi]   
  aux1 = list(r1)[hi:]+list(r1)[:hi]
  
  for a,b in zip(of1,of2):
   if a in aux2:
     aux2.remove(a)
   if b in aux1:
     aux1.remove(b)
     
  of2 = aux1[(ni-hi):]+ of2 + aux1[:(ni-hi)]
  of1 = aux2[(ni-hi):]+ of1 + aux2[:(ni-hi)]
  
  return of1,of2
 
def op_roulette(cost):
 c =  1./(1. + cost)
 p = c/c.sum()
 return pylab.where(p.cumsum() > pylab.rand())[0].min() 
 
def op_selection(cost,np):   
  
 perm = pylab.permutation(cost.size)
 j = [pylab.arange(i,cost.size,np) for i in range(np)]   
 idx = [op_roulette((cost[perm])[i]) for i in j]  
 
 return [perm[i] for i in idx]  


for kk in range(1):
 pylab.seed()
 P_crossover = 0.9# crossover probability
 P_mutation = 0.025 #probability of mutation 
 P_mutation2 = 0.25
 np = 4
 
 R = [pylab.permutation(ni) for i in range(nn)]
 cost = pylab.array([costfunc(r) for r in R]) 
 
 for iter in range(Niter):
    
  print(iter,cost.min(),cost.mean(),cost.std())
  
  Rnext = []
  cost_next = []
  
  for j,r in enumerate(R):
   # A candidata a ser propagada para a próxima geração é
   # inicialmente a solução atual r. Após crossover pode ser
   # substituida pela melhor filha se esta última for melhor que 
   # a solução atual. Por último, pode ser substituida pela sua mutação, 
   # caso esta última seja melhor.
   candidata = r
   fit_candidata = cost[j]
   
   if pylab.rand() < P_crossover:
    # seleciona np parentes para crossover
    # Realiza crossovers da solução atual com parentes selecionadas
    # gerando dois filhos por crossover
    rr = []
    
    for i in op_selection(cost,np):
     r1,r2 = op_crossover(r,R[i])
     rr = rr + [r1,r2]
     
    # custo das solucoes filhas
    cc = pylab.array([costfunc(i) for i in rr])
    # asssume a melhor solução filha como candidata 
    # caso esta seja melhor do que a atual
    if cc.min() < cost[j]:
      candidata = rr[cc.argmin()]
      fit_candidata = cc.min()
   # Aplica mutação à candidata 
   aux = op_mutation(candidata)
   fit_aux = costfunc(aux)
   # Propaga para a próxima geração o resultado da mutação
   # se este último for melhor do que a qualidade da candidata.
   if fit_aux < fit_candidata:
    candidata = aux
    fit_candidata = fit_aux
    
   Rnext.append(candidata)
   cost_next.append(fit_candidata)   
   
  R = Rnext.copy()
  cost = pylab.array(cost_next)
  
   