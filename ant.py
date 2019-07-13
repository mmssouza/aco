#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:10:52 2018

@author: msouza
"""

import scipy
import pylab
#from sklearn.preprocessing import scale
#from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
import pickle

def calc_MI(x, y, bins):
    c_xy = scipy.histogram2d(x, y, bins, normed = False)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

x = pickle.load(open("kimia99.data","rb"))
l = pickle.load(open("kimia99.labels","rb"))
x = pylab.log(x)[:,0:99:5]

ns,nf = x.shape

mi = 1. - scipy.zeros((nf,nf))

for i in range(nf):
 for j in range(nf):
   mi[i,j] = calc_MI(x[:,i],l,5)

#n_hf = 5 # hall of fame (top n_hf best solutions so far)
na = 20#nts number
alpha = .9# ant trend to follow collective trails
#beta =  0. # features independence influence
#gamma = .5  # accurancy vs dimensionality of solution tradeoff
rho = 0.15# pheromone evaporation rate
Q = 1.5#pheromone deposit gain
Epsilon = 0.01

# fitness matrix
tau = Epsilon*scipy.ones((nf,nf)) # pheromone matrix
fit = scipy.array([[scipy.rand(),0] for i in range(na)])

def fitness_func(features,xx,yy):
 #clf = GaussianNB()
 clf = KNeighborsClassifier(n_neighbors = 3,algorithm = 'kd_tree')
 score = cross_val_score(clf,xx[:,features],cv = StratifiedKFold(n_splits = 3),y = yy,scoring='accuracy')
 m,s = score.mean(),score.std()
# print(features.shape,ss)
 return [m,s]


def walk(orig,tau,nu = pylab.ones((nf,nf))): 
   p = scipy.array([(tau[orig,j]**alpha) * (nu[orig,j]**(1.-alpha)) for j in scipy.arange(nf)])
   p = (scipy.hstack(([0],p))/p.sum()).cumsum()
   dest = scipy.where(scipy.rand() > p)[0].argmax()
   return dest


def GeraSolucoes(nest,nattr):

  s = []
  # sorteia origem como funcao dos feromonios (orig)
  for i in scipy.arange(na):
   # sorteia número de atributos
   #nattr = walk(0,tau) + 1
#   print(n,end=" ")
   l = [nest]
   # Busca o primeiro atributo
   orig = nest
#   print(aux,end=" "
   #print(i,end = ' ')
   #print(nattr,end = ' ')
   #print(orig,end = ' ')
   # Busca dos n-1 demais atributos e coloca na lista l
   while len(l) < nattr:
    dest = walk(orig,tau,1/(mi+1))
    if dest not in l:
     l.append(dest)
     #print(dest,end = ' ')
     orig = dest
   #print() 
#   print(pylab.array(l)
   s.append(pylab.sort(l))
   
  return s

def AvaliaSolucoes(s,x,l):

   fit = scipy.array([fitness_func(xa,x,l) for xa in s])
   return(fit)

def AtualizaFeromonios(s): 
   
   ix = pylab.where(tau > Epsilon)
   tau[ix] = (1. - rho)*tau[ix] 
    
   for (i,xa) in enumerate(s):   
    for j,k in zip(xa[0:],xa[1:]):   
     tau[j,k] =  tau[j,k] + Q*fit[i,0]
    #tau[k,j] =  tau[k,j] + Q*fit[i,0]
  
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
 nest = 5
 nattr = 10
 for kk in scipy.arange(3000):
  sa =  GeraSolucoes(nest,nattr)
  fit = AvaliaSolucoes(sa,x,l)
  AtualizaFeromonios(sa)
  id_max = fit[:,0].argmax()
  id_min = fit[:,0].argmin()
  #print(kk,len(sa[id_max]),fit[id_max],fit[id_min],fit[:,0].mean())
  #print(pylab.sort(sa[id_max]))
  print(kk,fit[id_max][0],fit[id_min][0],fit[:,0].mean())
  #HF_Updt(hf,[fit[id_max][0],sa[id_max]])
 #print(hf)
