#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:10:52 2018

@author: msouza
"""

import scipy
import pylab
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
from numpy.random import permutation
from time import process_time


def calc_MI(x, y, bins):
    c_xy = scipy.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

x = pylab.loadtxt("madelon_train.data")
l = pylab.loadtxt("madelon_train.labels")

ns,nf = x.shape

#idx1 = permutation(scipy.arange(6000))[0:1000]
#idx2 = permutation(scipy.arange(5000))[0:2500]
#x = x[idx1]
#x = x[:,idx2]
#l = l[idx1]
idx1 = permutation(scipy.arange(ns))[0:1000]
idx2 = permutation(scipy.arange(nf))[0:50]
x = (x[idx1])[:,idx2]
l = l[idx1]
nf = x.shape[1]
print(x.shape)

mi = scipy.zeros((nf,nf))
for i in range(nf):
 for j in range(nf):
   mi[i,j] = calc_MI(x[:,i],x[:,j],10)

#n_hf = 5 # hall of fame (top n_hf best solutions so far)
na = 150 # ants number
alpha = 1.5 # ant trend to follow collective trails
beta = 0. # features independence influence
#gamma = .5  # accurancy vs dimensionality of solution tradeoff
rho = 0.15 # pheromone evaporation rate
Q = 10.  # pheromone deposit gain
# fitness matrix
tau = Q*scipy.ones((nf+2,nf)) # pheromone matrix
fit = scipy.array([[scipy.rand(),0] for i in range(na)])

def fitness_func(features,xx,yy):
 #clf = GaussianNB()
 clf = KNeighborsClassifier(n_neighbors = 3,algorithm = 'kd_tree')
 score = cross_val_score(clf,xx[:,features],cv = StratifiedKFold(n_splits = 10),y = yy,scoring='roc_auc')
 m,s = score.mean(),score.std()
# print(features.shape,ss)
 return [m,s]


def Sorteia(k,tau,nu = pylab.ones(nf)): 
   p = scipy.array([tau[k,j]**alpha * nu[j]**beta for j in scipy.arange(nf)])
   p = (scipy.hstack(([0],p))/p.sum()).cumsum()
   return scipy.where(scipy.rand() > p)[0].argmax()

#def Sorteia_n(k,tau): 
#   p = scipy.array([tau[k,j]**alpha for j in scipy.arange(nf)])
#   p = (scipy.hstack(([0],p))/p.sum()).cumsum()
#   return scipy.where(scipy.rand() > p)[0].argmax()

#def Sorteia_first(t):
#   p = scipy.array([t[1,j]**alpha for j in scipy.arange(nf)])
#   p = (scipy.hstack(([0],p))/p.sum()).cumsum()
#   return scipy.where(scipy.rand() > p)[0].argmax()

#def Sorteia_others(k,t):
#   p = scipy.array([(t[k,j]**alpha)*(1./(1e-3 + mi[k,j])**beta) for j in scipy.arange(nf)])
#   p = (scipy.hstack(([0],p))/p.sum()).cumsum()
#   return scipy.where(scipy.rand() > p)[0].argmax()

def GeraSolucoes():

  s = []
  # sorteia origem como funcao dos feromonios (orig)
  for i in scipy.arange(na):
   # sorteia número de atributos
   n = Sorteia(0,tau) + 1
   # Busca o primeiro atributo
   aux = Sorteia(1,tau)
   # Busca dos n-1 demais atributos e coloca na lista l
   l = []
   for k in scipy.arange(n-1):
    l.append(aux)
    aux = Sorteia(aux,tau[2:,:],1./mi[aux,:])
   # elimina repetidos e identifica os atributos não pertences a l
   s1 = set(l)
   l = list(s1)
   u = list(set(range(nf)) - s1)
   # Adiciona aleatóriamente, no lugar dos atributos repetidos
   # aqueles que não pertencem a l até que a cardinalidade
   # atinja n
   while (len(l) < n):
     u = list(permutation(u))
     l.append(u.pop())
   s.append(scipy.sort(l))
  return s

def AvaliaSolucoes(s,x,l):

   fit = scipy.array([fitness_func(xa,x,l) for xa in s])
   return(fit)

def AtualizaFeromonios(s,tau):

   for (i,xa) in enumerate(s):
    sz = xa.shape[0]
    gain_fit = fit[i,0]    # pheromone ant gains due to classification accurancy
    #gain_length = (1./((float(sz)/float(nf))+1.)**2)  # pheromone ant gains due to number of features
    #gain = Q*(gain_fit**gamma)*(gain_length**(1-gamma))  # accurancy vs number of features tradeoff
    gain = Q*gain_fit  # accurancy vs number of features tradeoff
    tau[0,sz-1] = tau[0,sz-1] + gain
    tau_s = tau[1:,:]
    k = xa[0]
    for j in xa[1:]:
     tau_s[k,j] =  tau_s[k,j] + gain
     k = j
   tau =  (1. - rho)*tau

def HF_Updt(hf,x):
  if len(hf) < n_hf:
   hf.append(x)
   hf.sort(reverse = True)
   return

  # Hall of fame
  aux = scipy.where(x[0] > scipy.array([i[0] for i in hf]))
  if len(aux[0]) != 0:
   hf.insert(aux[0].min(),x)
   hf.pop()

if __name__ == '__main__':
 hf = []
 for kk in scipy.arange(3000):
  tt = process_time()
  sa =  GeraSolucoes()
  fit = AvaliaSolucoes(sa,x,l)
  AtualizaFeromonios(sa,tau)
  id_max = fit[:,0].argmax()
  id_min = fit[:,0].argmin()
  print(kk,process_time() - tt,len(sa[id_max]),fit[id_max],fit[id_min],fit[:,0].mean())
  print(sa[id_max])
  #HF_Updt(hf,[fit[id_max][0],sa[id_max]])
 #print(hf)
