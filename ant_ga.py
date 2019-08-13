#!/usr/bin/python3 -u
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:10:52 2018

@author: msouza
"""

import scipy
import configparser
import aco
import ga_class
from scipy.spatial.distance import pdist,squareform

config = configparser.ConfigParser()
config.read("parameters.conf")

parms = config['default']

infile = parms['infile']
Niter = int(parms['Niter'])
Ncpu = int(parms['Ncpu'])

parms = config['aco']

na = int(parms['na']) # ants number
alpha = float(parms['alpha']) # ant trend to follow collective trails
beta = float(parms['beta']) # ant trend to follow individual shortest path
rho = float(parms['rho']) # pheromone evaporation rate
Q = float(parms['Q']) # pheromone deposit gain
Epsilon = float(parms['Epsilon']) # smallest pheromone deposit
hf_sz = int(parms['hf_sz']) # Hall of fame size

parms = config['ga']

Npop = int(parms['Npop']) # ants number
Pc = float(parms['P_crossover']) # ants number
Pm = float(parms['P_mutation']) # ants number

raw_data = scipy.loadtxt(infile,skiprows = 6,usecols = (1,2),comments='E')

dist = squareform(pdist(raw_data))

fit = lambda x: scipy.sum([dist[i,j] for i,j in zip(x,x[1:])])

   
if __name__ == '__main__':
  ant = aco.aco(na,alpha,beta,rho,Q,Epsilon,dist,fit,hf_sz) 
  ga = ga_class.ga(Npop,2,Pc,Pm,dist,fit)
  ga.Import(ant.sa)
  feed = True
  count = 0
  
  for kk in range(Niter):   
   ant.Iter()
   
   if feed and (kk % (2*hf_sz) == 0) and (kk != 0):
    ga.Import(ant.sa+ant.hf)
    count = count+1
    if count > 0.5*Niter/(2*hf_sz):
        feed= False
        
   ga.Iter()
   
   print("{:4d} {:5.0f} {:5.0f} {:5.0f} {:5.0f}".format(kk,ant.fitness.min(),ant.fitness.max(),ant.fitness.mean(),ant.fitness.std()))  
   print("{:4d} {:5.0f} {:5.0f} {:5.0f} {:5.0f}".format(kk,ga.fitness.min(),ga.fitness.max(),ga.fitness.mean(),ga.fitness.std()))
  
  print(ga.fitness.min(),ga.pop[ga.fitness.argmin()])
  print(ant.fitness.min(),ant.sa[ant.fitness.argmin()])
    