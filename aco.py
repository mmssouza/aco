#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:04:30 2019

@author: msouza
"""
import scipy

class aco:
    def __init__(self,na,alpha,beta,rho,Q,Eps,dmat,fitfunc,hf_sz):
       self.na = na
       self.alpha = alpha
       self.beta = beta
       self.rho = rho
       self.dmat = dmat
       self.Eps = Eps
       self.n = self.dmat.shape[0]
       self.Q = Q
       self.fit = fitfunc
       self.tau =  self.Eps*scipy.ones((self.n,self.n))
       self.sa = [self.GeraSolucao() for i in range(self.na)]
       self.fitness = self.AvaliaSolucoes(self.sa)
       self.hf = []
       self.hf_sz = hf_sz
       
    def Sorteia(self,k,U,nu):
        p = scipy.zeros(self.n)
        
        for j in U:
            p[j] = (self.tau[k,j]**self.alpha)*(self.Q/nu[k,j])**self.beta    
            
        p = scipy.hstack(([0],p/p.sum())).cumsum()
        return scipy.where(scipy.rand() > p)[0].argmax()
    
    def GeraSolucao(self):
       U = list(range(self.n))
       orig = 0
       l = [orig]
       U.remove(orig)

       while len(U) > 0:
           aux = self.Sorteia(orig,U,self.dmat)
           l.append(aux)
           U.remove(aux)
           orig = aux
       return l
    
    def HF_Updt(self,x):

        x_fit = int(self.fit(x))
        hf_fit = self.AvaliaSolucoes(self.hf).astype(int)
   
        if len(self.hf) < self.hf_sz:
            self.hf.append(x)
            self.hf.sort(reverse = False)
        else:
            # Hall of fame
            aux = scipy.where(x_fit <= hf_fit)
            if len(aux[0]) != 0:
                i = aux[0].min()
                if x_fit != hf_fit[i]:
                    self.hf.insert(i,x)
                    self.hf.pop()
                    
    def Import(self,p):
        idx = scipy.random.permutation(self.na)
        for i,x in zip(idx,p):
         fit_x = self.fit(x)
         if fit_x < self.fitness[i]:
             self.sa[i] = x
             self.fitness[i] = fit_x
             
    def AvaliaSolucoes(self,s): 
        return scipy.array([self.fit(x) for x in s])

    def AtualizaFeromonios(self):
        for (i,xa) in enumerate(self.sa):
            for j,k in zip(xa,xa[1:]):
                self.tau[j,k] = self.tau[j,k] + self.Q/float(self.fitness[i]) 
        idx = scipy.where(self.tau > self.Eps)
        self.tau[idx] = self.rho*self.tau[idx]
        
    def Iter(self):
         self.HF_Updt(self.sa[self.fitness.argmin()])
         self.AtualizaFeromonios()
         self.sa = [self.GeraSolucao() for i in range(self.na)] 
         self.fitness = self.AvaliaSolucoes(self.sa)

      