# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 13:28:48 2018

@author: Piotr
"""
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import numpy as np

class ploter:
    
    def __init__(self, fig_in=0, ax1_in=0, ax2_in=0):
        self.td = {'set':1,'p_name':'DMshape_c1', 'vals':['rect','gauss'], 'label':'sh'}
        self.ld = {'set':1,'p_name':'Tservo_c1', 'vals':[1,10], 'from':None, 'to':None, 'label':'Ts'}
        self.xd = {'p_name':'DMdur_c1', 'vals':[1,2,3,4,5,6], 'from':None, 'to':None, 'label':'DMdur [s]'}
        self.y1d = {'p_name':'maxcor_out', 'label':'sensitivity'}
        self.y2d = {'set':0,'p_name':'SNRcor_out', 'label':'SNR'}
        #------------------------------------------------------------------
        self.color_tab = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.color_index = 0
        self.type_tab = ['-', '--', '-.', ':']
        self.type_index = 0
        self.markersik=''
        self.grmod='none'
        self.rescalx=1
        #self.linestyle = '-'
        
        if fig_in==0:
            self.fig, self.ax1 = plt.subplots()
            if self.y2d['set']!=0:
                self.ax2 = self.ax1.twinx()
        else:
            self.fig=fig_in
            self.ax1=ax1_in
            self.ax2=ax2_in
        self.label=[]

    def singleplot(self,f, label='', color='b', linestyle='-'):
        #wybierz x, y1 i posortuj
      #  fm = f.groupby([self.xd['p_name']]).mean()
        #linestyle = self.linestyle
        if self.grmod=='sort':
            xys = f[[self.xd['p_name'],self.y1d['p_name']]].sort_values(by=[self.xd['p_name']]).values
        elif self.grmod=='mean':
            xys = f[[self.xd['p_name'],self.y1d['p_name']]].groupby(self.xd['p_name']).mean().reset_index().values
        elif self.grmod=='min':
            xys = f[[self.xd['p_name'],self.y1d['p_name']]].groupby(self.xd['p_name']).min().reset_index().values
        elif self.grmod=='std':
            xys = f[[self.xd['p_name'],self.y1d['p_name']]].groupby(self.xd['p_name']).std().reset_index().values
        else:
            xys = f[[self.xd['p_name'],self.y1d['p_name']]].values
        
        if self.grmod=='scat':
            self.ax1.scatter(xys[:,0]*self.rescalx, xys[:,1],  
                     label=label, 
                     color=color,
                     #linestyle=linestyle,
                     #linewidth=2,
                     marker='.')
        
        else:
            self.ax1.plot(xys[:,0]*self.rescalx, xys[:,1],  
                     label=label, 
                     color=color,
                     linestyle=linestyle,
                     linewidth=2,
                     marker=self.markersik)
            
        self.ax1.set_xlabel(self.xd['label'])
        self.ax1.set_ylabel(self.y1d['label'])
        self.ax1.grid(b=1,which='both')
        
        if self.y2d['set'] != 0:
       
            #wybierz x, y1 i posortuj
            xys = f[[self.xd['p_name'],self.y2d['p_name']]].sort_values(by=[self.xd['p_name']]).values
            self.ax2.plot(xys[:,0]*self.rescalx, xys[:,1], 
                     color=color,
                     linestyle=linestyle,
                     linewidth=0.5)
            self.ax2.set_ylabel(self.y2d['label'])   
            self.fig.tight_layout()
    
    def lplot(self,f, linestyle='-', label=''):
        self.color_index=0
        if self.ld['set'] !=0:
            for li in self.ld['vals']:      
                lf = f[ ( f[self.ld['p_name']] == li ) ]
                self.singleplot(lf, 
                           label= label + self.ld['label']+'='+str(li),
                           color = self.color_tab[self.color_index],
                           linestyle=linestyle )
                self.color_index = self.color_index+1
        else:
            self.singleplot(f, label='', color='b', linestyle=linestyle)
            
    def tplot(self,f):
        self.type_index=0
        if self.td['set'] != 0:
            for ti in self.td['vals']:
                tf = f[ ( f[self.td['p_name']] == ti ) ]
                self.lplot(tf,linestyle=self.type_tab[self.type_index],
                      label = self.td['label']+'='+str(ti)+' ')
                self.type_index=self.type_index+1
        else:
            self.lplot(f, linestyle='-')
    
    def show(self):
        plt.subplots_adjust(top=0.8)
        #self.fig.show()
        plt.show()
        
#-------------------------------------------
def to_base(datdic, file='base.pkl'):
    dnew={}
    for i in datdic.keys():
        dic = datdic[i]
        for k in dic.keys():
            dnew[k+'_'+i]=dic[k]
    dnew['TimeStamp']=pd.Timestamp.now()
    ser = pd.Series(dnew)  
    if os.path.isfile(file):
        pdata = pd.read_pickle(file)
        pdata = pdata.append(ser, ignore_index=True)
    else:
        pdata = pd.DataFrame()
        pdata = pdata.append(ser, ignore_index=True)
    pdata.to_pickle(file)

#------------------------------------------
def wyrownaj(t1, d1, t2, d2):
    t_total_1 = t1[-1]-t1[0]
    t_total_2 = t2[-1]-t1[0]
    len_1 = len(t1)
    len_2 = len(t2)
    step1 = t_total_1/len_1
    step2 = t_total_2/len_2
    if step1>step2:
        tmas = t2.copy()
        dmas = d2.copy()
        tslv = t1.copy()
        dslv = d1.copy()
    else:
        tmas = t1.copy()
        dmas = d1.copy()
        tslv = t2.copy()
        dslv = d2.copy()
    
    dnew = dmas.copy()
    lenslv = len(tslv)
    islv = 0
    for imas in range(len(tmas)):
        islv = islv + 1
        if islv >= lenslv :
            islv = islv -1
        if tmas[imas]<tslv[islv]:
            islv = islv - 1 
        dnew[imas]=dslv[islv]
    
    return tmas, dmas, dnew

def densgrid(t,d,size=10,step=0.01):
    tn = np.arange(0,size, step)
    i=0
    ld=d[i]
    rd=d[i+1]
    lt=t[i]
    rt=t[i+1] 
    slope = (rd-ld)/(rt-lt)
    dn=np.array([])
    for ti in tn:
        if ti>rt:
            i=i+1
            ld=d[i]
            rd=d[i+1]
            lt=t[i]
            rt=t[i+1]
            slope = (rd-ld)/(rt-lt)
        dn=np.append(dn, ld+slope*(ti-lt))
    return tn, dn

