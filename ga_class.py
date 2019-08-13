#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:03:24 2019

@author: msouza
"""

import scipy
 
class ga:
    def __init__(self,N,NP,P_crossover,P_mutation,dmat,fit_func):
      self.Npop = N
      self.NP = NP
      self.P_crossover = P_crossover
      self.P_mutation = P_mutation/10
      self.P_mutation2 = P_mutation
      self.Ni = dmat.shape[0]
      self.fit = fit_func
      self.pop =  [scipy.random.permutation(self.Ni) for i in range(self.Npop)]
      self.fitness = scipy.array([self.fit(r) for r in self.pop])  
       
    def op_mutation2(self,r):
        
      if scipy.rand() < self.P_mutation:
       i,j = None,None  
       
       while True: 
        i,j = scipy.random.randint(low = 1,high = self.Ni-1,size = 2) 
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
    
    def op_mutation(self,r):
        
      for i,x in enumerate(r):
       if scipy.rand() < self.P_mutation:
         if i > 0:
          aux = r[i-1]
          r[i-1] = x
          r[i] = aux
         else:
          aux = r[-1]
          r[-1] = x
          r[i] = aux
         
      return r
     
    def op_crossover(self,r1,r2):
      
      while True: 
       i,j = scipy.random.randint(low = 1,high = self.Ni-1,size = 2) 
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
         
      of2 = aux1[(self.Ni-hi):]+ of2 + aux1[:(self.Ni-hi)]
      of1 = aux2[(self.Ni-hi):]+ of1 + aux2[:(self.Ni-hi)]
      
      return of1,of2
  
    def op_roulette(self,cost):
      c =  1./(1. + cost)
      p = c/c.sum()
      return scipy.where(p.cumsum() > scipy.rand())[0].min() 
 
    def op_selection(self):   
        
      perm = scipy.random.permutation(self.fitness.size)
      j = [scipy.arange(i,self.fitness.size,self.NP) for i in range(self.NP)]   
      idx = [self.op_roulette((self.fitness[perm])[i]) for i in j]  
      
      return [perm[i] for i in idx]  
  
    #def op_selection(self):   
    #  
    # idx = scipy.random.permutation(self.Npop)
   # 
   #  half = int(idx.size/2)
   # 
   #  idx1,idx2 = idx[0:half],idx[half:idx.size]
     
   #  c1 = scipy.array(self.fitness)[idx1] 
   #  c1 =  1./(1. + c1)
    
   #  c2 = scipy.array(self.fitness)[idx2]  
   #  c2 =  1./(1. + c2)
     
   #  p1 = c1/c1.sum()
   #  p2 = c2/c2.sum()
    
   #  i1  = scipy.where(p1.cumsum() > scipy.rand())[0].min()
   #  i2 = scipy.where(p2.cumsum() > scipy.rand())[0].min()
    
   #  return idx1[i1],idx2[i2]  
    
    def Import(self,p):
        for i,r in zip(scipy.random.permutation(self.Npop),p):
         fit_r = self.fit(r)
         if  fit_r < self.fitness[i]:
             self.pop[i] = r
             self.fitness[i] = fit_r
            
    def Iter(self):
      
      pop_next = []
      fit_next = []
      
      for j,r in enumerate(self.pop):
       # A candidata a ser propagada para a próxima geração é
       # inicialmente a solução atual r. Após crossover pode ser
       # substituida pela melhor filha se esta última for melhor que 
       # a solução atual. Por último, pode ser substituida pela sua mutação, 
       # caso esta última seja melhor.
       candidata = r
       fit_candidata = self.fitness[j]
       
       if scipy.rand() < self.P_crossover:
        # seleciona dois parentes para crossover 
        #i1,i2 = self.op_selection() 
        
        rr = []
        idx = self.op_selection()
        
        for ii,i in enumerate(idx[:self.NP-1]):
         for k in idx[ii+1:]:
          r1,r2 = self.op_crossover(self.pop[i],self.pop[k])
          rr = rr + [r1,r2]
     
        # custo das solucoes filhas
        cc = scipy.array([self.fit(i) for i in rr])
        # Realiza crossovers da solução atual com as parentes selecionadas
        # gerando quatro filhos
        # asssume a melhor solução filha como candidata 
        # caso esta seja melhor do que a atual
        #if cc.min() < self.fitness[j]:
        candidata = rr[cc.argmin()]
        fit_candidata = cc.min()
        #candidata = scipy.random.permutation(rr)[0]
          
       # Aplica mutação à candidata 
       aux = self.op_mutation2(candidata) 
       # Propaga para a próxima geração o resultado da mutação
       # se este último for melhor do que a qualidade da candidata.
       fit_aux = self.fit(aux)
       
       faux = scipy.array([fit_aux,fit_candidata,self.fitness[j]])
       
       fit_next.append(faux.min())
       pop_next.append(scipy.array([aux,candidata,r])[faux.argmin()])
            
      self.fitness = scipy.array(fit_next)
      self.pop = pop_next.copy()
      
      
       