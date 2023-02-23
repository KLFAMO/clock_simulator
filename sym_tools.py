  # -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:00:15 2018

@author: Piotr Morzyński
"""

import numpy as np
import matplotlib.pyplot as plt
import tools as tls
from scipy.optimize import curve_fit

#functions==============================================
def senfun(t,Tint,D_rad, gtype='rabi'):
    """ generate sensitivity function
        for Rabi spectroscopy, 
        start at t=0 stop at t=Tint
    """
    if gtype=='rabi':
        DTD = Tint*D_rad/np.pi
        te = np.pi/2.+np.arctan(DTD)
        Omt = np.pi*np.sqrt(1+np.power(DTD,2))
        Om1 = Omt*t/Tint
        Om2 = Omt*(Tint-t)/Tint
        g1 = np.power(np.sin(te),2)*np.cos(te)
        g2 = ( (1-np.cos(Om2))*np.sin(Om1) +
               (1-np.cos(Om1))*np.sin(Om2)  )
        return g1*g2
    
    elif gtype=='uni':
        return np.double(-0.60386)
    
    else:
        return 0


#========================================================
def P_bloch(Tint,Delta_Hz):
    """ simulate Bloch equations
        return excitation prob after interrogation
    """
    ee_o=0; ge_o=0
    ee_n=0; ge_n=0
    t_tab = np.arange(0,Tint, Tint/1000000)
    dt = (t_tab[-1]-t_tab[0])/(len(t_tab)-1)
    Om = np.pi/Tint
    P_tab=[]
    for i in range(len(t_tab)):
        D = (Delta_Hz*2*np.pi)    
        ee_n = ee_o - Om * ge_o.imag  * dt
        ge_n = ge_o +1j*(-1*D*ge_o + Om*(ee_o - 0.5 ) ) * dt
        ee_o = ee_n
        ge_o = ge_n    
        P_tab.append(ee_n.real)
#    plt.plot(t_tab,P_tab)
#    plt.grid()
#    plt.show()
    return ee_n.real

#=======================================================
def P_sf_1(Tint, Delta_rad, subn, gtype='rabi'):
    dt = Tint/subn
    t_tab = np.arange(0,Tint, dt)
    g_tab = np.array([senfun(x,Tint, Delta_rad, gtype=gtype) for x in t_tab])
    #print('**********',np.max(g_tab), np.min(g_tab),'**************')
    return t_tab, g_tab, dt
    
#=======================================================
def P_sf_2(Tint, EDelta_rad_tab, t_tab, g_tab):
    suma=0  
    for i in range(len(t_tab)):
        suma=suma+EDelta_rad_tab[i]*g_tab[i]
    return 0.5*suma*Tint/len(t_tab)+0.5

#=======================================================
def calc_param(s):
    ns = s.copy()
    #some parameters calculations--------------
    if not 'servocycles' in ns.keys():
        ns['servocycles']=2
    if ( not 'Tcycle' in ns.keys() )  or ( ns['Tcycle'] == 'c' ):
        ns['Tcycle'] = ns['Tservocycle']/ns['servocycles']
    if ( not 'Tint' in ns.keys() )  or  ( ns['Tint'] == 'c' ):
        ns['Tint'] = ns['Tcycle']*ns['w']

    ns['DM_timeshift'] = -ns['DMdur']/2-ns['DM_synctimeshift']
    ns['FWHM_Hz'] = 0.399343*2/ns['Tint'] #w Hz
    ns['ser_det_Hz'] = ns['FWHM_Hz']/2
    ns['ser_det_rad'] = 2*np.pi*ns['ser_det_Hz']
    ns['dndD_Hz'] = 0.60386*ns['Tint']*2*np.pi
    ns['DM_lineshift_rad']=ns['DM_lineshift_Hz']*2*np.pi
    ns['subinter_dt'] = ns['Tint']/ns['subinter_points']
    ns['k'] = ns['dndD_Hz']
    if not 'Tservocycle' in ns.keys():
        ns['Tservocycle']=ns['servocycles']*ns['Tcycle']
    if ns['I']=='c':
        ns['I'] = (1-np.exp(ns['Tservocycle']/
                      (-1*ns['Tservo']) ))/(ns['k']*ns['servocycles']/2)
    if ns['DMshape']=='sin':
        ns['DMperiod']=ns['DMdur']
    if ns['DMperiod'] == 'c':
        ns['DMperiod'] = np.ceil(ns['DMperiod_factor']*ns['Tservo']+
                               5*ns['DMdur'])+ns['DMduradd']
        if ns['DMperiod']<10:
            ns['DMperiod'] = 10 + ns['DMduradd']
    ns['half_DMperiod'] = ns['DMperiod']/2
    if ns['Tmeas']=='c':
        if ns['DMduradd']!=0:
            ns['Tmeas']= ns['DMperiod']*2./ns['DMduradd']
        else:
            ns['Tmeas']= ns['DMperiod']*100
    ns['cycles'] = int(ns['Tmeas']/ns['Tcycle'])
    return ns

#=======================================================
def rect(t, shift, dur):
    if t>shift-dur/2. and t<dur/2.+shift:
        return 1
    return 0
#=======================================================
#=========================================================
def gauss(t, shift, dur):
    durp = 0.5*dur/0.8862
    return np.exp(-np.power(t - shift, 2.) / (2 * np.power(durp, 2.)))

#==========================================================
def DMfun(t, shift, dur, shape='rect'):
    if shape=='gauss':
        return gauss(t, shift, dur)
    elif shape=='rect':
        return rect(t, shift, dur)
    else:
        return np.sin(2*np.pi*(t-shift)/dur)
#=========================================================
def mplot(t_tab, zm, zm_names, figname='', figsize=[8, 14], 
          fignum=None, axs=[], s=[], ev=1):
    m = len(zm)
    if len(axs)<1:
        fignum, axs = plt.subplots( nrows=m,
                    figsize=figsize, sharex=True)
    for i in range(1,m+1) :
        axs[i-1].plot(t_tab[::ev], zm[i-1][::ev], '.-', linewidth=0.5,
           label=str(s['name']))
        axs[i-1].grid(b=1, which='both', axis='both')
        plt.ylabel(zm_names[i-1])
        plt.legend()

    return fignum, axs
#========================================================
def symclk(s, is_prob_noise=0):
 
    ns = calc_param(s)
    # some variables and tables for speedup
    t_tab, g_tab_p, dt = P_sf_1(ns['Tint'], ns['ser_det_rad'],
                                subn=ns['subinter_points'],
                                gtype=ns['gtype'])  
    #cavity frequency in time
    f_cav_tab = np.array([0.]*ns['cycles'])
    #create empty tables
    P_tab=[]
    overlap_tab=[]
    t_tot_tab = []
    f_aom_tab = []
    err_tab = []
    err_int_tab = []
    f_L_tab = []
    f_atom_tab=[]
    #initial parameters
    t_to_DM = ns['DM_timeshift']+ns['DMperiod']
    n=0
    side=1  
    scycle=1
    t_tot = 0
    f_aom = 0
    err = 0
    err_int = 0
    #Pold=0
    #simulation loop-----------------------------------
    while n < ns['cycles']:
        #calc new position to DM-------
        t_to_DM = t_to_DM + ns['Tcycle']
        if t_to_DM >= ns['half_DMperiod']:
            t_to_DM = t_to_DM - ns['DMperiod']
        #calc probability-------------
        #create event tab
        Ev_tab=np.array([
                DMfun(x,t_to_DM, ns['DMdur'], shape=ns['DMshape']) 
                        *ns['DM_lineshift_rad'] 
                        for x in t_tab])
        overlap_tab.append( np.sum(Ev_tab)/(ns['subinter_points']*2*np.pi) )
        #calc excitation probability
        P = P_sf_2(ns['Tint'], -Ev_tab*side+(
                f_cav_tab[n]+f_aom)*2*np.pi*side, t_tab, g_tab_p)

        #controller-------------------
        if side < 0 :
            err = err-P
            if scycle>=ns['servocycles']:
                #add noise-------------------
                if is_prob_noise:
                    err=err+np.random.normal(0, ns['prob_noise_amp'])
                err_int = err_int + err
                f_aom = f_aom + err * ns['I']
                scycle=0
                err=0
        else:
            err = err + P
        #add to tables ---------------------------
        f_atom_tab.append(f_cav_tab[n]+f_aom)
        err_tab.append(err)
        err_int_tab.append(err_int)
        P_tab.append(P)
        t_tot_tab.append(t_tot)
        f_aom_tab.append(f_aom)
        f_L_tab.append(f_cav_tab[n]+f_aom)
        #change state ----------------------------
        t_tot = t_tot+ns['Tcycle']
        n=n+1
        side = -1*side
        scycle=scycle+1
        
       
    return np.array(t_tot_tab), np.array(f_aom_tab), np.array(overlap_tab), ns
#--------------------------------------------------------------
def symcor(s1, s2, plot=0):
    sout={}
    maxcor = 0
    newnum=None; ax=[]
    s1 = calc_param(s1)
    s2 = calc_param(s2)
    #simulate first clock and plot
    t_tab1, aom_tab1, DM_tab1, ns1 = symclk(s1)
    #t_tab1n, aom_tab1n, DM_tab1n, ns1n = symclk(s1, is_prob_noise=1)
    if plot>0:
        newnum, ax = mplot(t_tab1, 
                           (aom_tab1, DM_tab1, DM_tab1-aom_tab1), 
                           ('f_aom', 'f_DM',   'difference'), 
               fignum=newnum, axs = ax, s=s1)
    #simulate second clock and plot
    t_tab2, aom_tab2, DM_tab2, ns2 = symclk(s2)
   # t_tab2n, aom_tab2n, DM_tab2n, ns2n = symclk(s2, is_prob_noise=1)
    if plot>0:
        newnum, ax = mplot(t_tab2, 
                           (aom_tab2, DM_tab2, DM_tab2-aom_tab2), 
                           ('f_aom', 'f_DM',  'difference'), 
               fignum=newnum, axs = ax, s=s2)
    #analyse------------
      #allign grids and correlate
    t_time, aom_tab1al, aom_tab2al = tls.wyrownaj(t_tab1,aom_tab1, t_tab2, aom_tab2)
    size = min(t_tab1[-2],t_tab2[-2])
    t_time, aom_tab1al = tls.densgrid(t_tab1,aom_tab1,size=size)
    t_time, aom_tab2al = tls.densgrid(t_tab2,aom_tab2,size=size)
    cor = np.correlate(aom_tab1al,aom_tab2al,'full')
    #plt.plot(cor)
    #plt.show()
      #same with noise
#    t_time, aom_tab1al, aom_tab2al = tls.wyrownaj(t_tab1n,aom_tab1n, t_tab2n, aom_tab2n)
#    corn = np.correlate(aom_tab1al,aom_tab2al,'full')   
    # DM autocorrelation --------------
    t_tab = np.arange(-0.5*s1['DMperiod'],0.5*s1['DMperiod'],1e-4)
    shape_tab = np.array([DMfun(x,0,s1['DMdur'],s1['DMshape'])*s1['DM_lineshift_Hz']
                                for x in t_tab ])
    shape_sqr_integral = np.sum(np.power(shape_tab,2))*1e-4
    DMnorm = shape_sqr_integral #*s1['Tmeas']/s1['DMperiod']
    ns1['DMmaxcor']=DMnorm
    #-------------
    shape_tab2 = np.array([DMfun(x,0,s2['DMdur'],s2['DMshape'])*s2['DM_lineshift_Hz']
                                for x in t_tab ])
    shape_sqr_integral2 = np.sum(np.power(shape_tab2,2))*1e-4
    DMnorm2 = shape_sqr_integral2#*s2['Tmeas']/s2['DMperiod']
    ns2['DMmaxcor'] = DMnorm2
    #chceck DM
    ns1['DMmin']=DMfun(0.5*s1['DMperiod'],0,s1['DMdur'])
    ns2['DMmin']=DMfun(0.5*s2['DMperiod'],0,s2['DMdur'])
    # output calculations ---------------------------
    dT=(t_time[-1]-t_time[0])/(len(t_time)-1)
    maxcor = np.max(cor)*dT/DMnorm #correlation amplitude without noise
    sout['maxcor']=maxcor
    #sout['stdcor']=np.std((corn-cor)*Tcycle/DMnorm) #std of correlation noise
    #sout['SNRcor']=sout['maxcor']/sout['stdcor']
    sout['dT']=dT
    print(sout)
   
    return ns1, ns2, sout


def symosc(s1):
    sout={}
    s1 = calc_param(s1)
    t_tab1, aom_tab1, DM_tab1, ns1 = symclk(s1)
    t_tab1=t_tab1[::2].copy()
    aom_tab1=aom_tab1[::2].copy()
    def my_sin(t, amp, phase):
        return amp*np.sin(2*np.pi*(1/ns1['DMdur'])*t+phase)
    p0 = [-min(aom_tab1), 0]
    popt, pcov = curve_fit(my_sin, t_tab1, aom_tab1, p0=p0)
    print('fit: ',popt)
#    plt.plot(t_tab1, aom_tab1, '.')
#    plt.plot(t_tab1, my_sin(t_tab1, popt[0], popt[1]))
#    plt.show()
    sout['ampl']=popt[0]
    sout['phase']=popt[1]
    sout['sens']=sout['ampl']/ns1['DM_lineshift_Hz']
    return ns1, sout
#--------------------------------------------------------------

def symcor_with_noise(s1, s2, plot=0):
    sout={}
    maxcor = 0
    newnum=None; ax=[]
    s1 = calc_param(s1)
    s2 = calc_param(s2)
    #simulate first clock and plot
    t_tab1, aom_tab1, DM_tab1, ns1 = symclk(s1)
    t_tab1n, aom_tab1n, DM_tab1n, ns1n = symclk(s1, is_prob_noise=1)
    if plot>0:
        newnum, ax = mplot(t_tab1, 
                           (aom_tab1, DM_tab1, DM_tab1-aom_tab1), 
                           ('f_aom', 'f_DM',   'difference'), 
               fignum=newnum, axs = ax, s=s1)
    #simulate second clock and plot
    t_tab2, aom_tab2, DM_tab2, ns2 = symclk(s2)
    t_tab2n, aom_tab2n, DM_tab2n, ns2n = symclk(s2, is_prob_noise=1)
    if plot>0:
        newnum, ax = mplot(t_tab2, 
                           (aom_tab2, DM_tab2, DM_tab2-aom_tab2), 
                           ('f_aom', 'f_DM',  'difference'), 
               fignum=newnum, axs = ax, s=s2)
    #analyse------------
      #allign grids and correlate
    #t_time, aom_tab1al, aom_tab2al = tls.wyrownaj(t_tab1,aom_tab1, t_tab2, aom_tab2)
    size = min(t_tab1[-2],t_tab2[-2])
    t_time, aom_tab1al = tls.densgrid(t_tab1,aom_tab1,size=size)
    t_time, aom_tab2al = tls.densgrid(t_tab2,aom_tab2,size=size)
    cor = np.correlate(aom_tab1al,aom_tab2al,'full')
    #plt.plot(cor)
    #plt.show()
      #same with noise
   # t_time, aom_tab1al, aom_tab2al = tls.densgrid(t_tab1n,aom_tab1n, t_tab2n, aom_tab2n)
    t_time, aom_tab1al = tls.densgrid(t_tab1n,aom_tab1n,size=size)
    t_time, aom_tab2al = tls.densgrid(t_tab2n,aom_tab2n,size=size)
    corn = np.correlate(aom_tab1al,aom_tab2al,'full')   
    #plt.plot(corn)
    #plt.show()
    # DM autocorrelation --------------
    t_tab = np.arange(-0.5*s1['DMperiod'],0.5*s1['DMperiod'],1e-4)
    shape_tab = np.array([DMfun(x,0,s1['DMdur'],s1['DMshape'])*s1['DM_lineshift_Hz']
                                for x in t_tab ])
    shape_sqr_integral = np.sum(np.power(shape_tab,2))*1e-4
    DMnorm = shape_sqr_integral #*s1['Tmeas']/s1['DMperiod']
    ns1['DMmaxcor']=DMnorm
    #-------------


    shape_tab2 = np.array([DMfun(x,0,s2['DMdur'],s2['DMshape'])*s2['DM_lineshift_Hz']
                                for x in t_tab ])
    shape_sqr_integral2 = np.sum(np.power(shape_tab2,2))*1e-4
    DMnorm2 = shape_sqr_integral2#*s2['Tmeas']/s2['DMperiod']
    ns2['DMmaxcor'] = DMnorm2
    #chceck DM
    ns1['DMmin']=DMfun(0.5*s1['DMperiod'],0,s1['DMdur'])
    ns2['DMmin']=DMfun(0.5*s2['DMperiod'],0,s2['DMdur'])
    # output calculations ---------------------------
    dT=(t_time[-1]-t_time[0])/(len(t_time)-1)
    maxcor = np.max(cor)*dT/DMnorm #correlation amplitude without noise
    sout['maxcor']=maxcor
    sout['corNA_at0_noise']=corn[int(np.round(len(corn)/2))]/len(corn)
    std = np.std((corn-cor)*dT/DMnorm)
    #print('std: ', std)
    sout['stdcor']=std #std of correlation noise
    sout['stdcor_ad'] = np.std(corn)
    sout['SNRcor']=sout['maxcor']/sout['stdcor']
    sout['dT']=dT
    print(sout)
   
    return ns1, ns2, sout
#--------------------------------------------------------------
def multiscan(param_dicts, scans, depth=0, base_name='base2.pkl'):
    
    scan = scans[depth]
    for i in scan['vals']:
        #set parameters-------------
        for j in scan['which']:
            param_dicts[j][scan['p_name']]=i
        #run or scan deeper---------
        if depth < len(scans)-1:  #go deeper
            multiscan(param_dicts, scans, depth=depth+1, base_name=base_name)
        else: #run simultion
            # print current parameters
            print('************* ')
            for pi in scans:
                print(pi['p_name']+' = '+
                     str([param_dicts[di][pi['p_name']] 
                         for di in [0,1]])  )
            #run simulation
            ns1,ns2,nsout = symcor(param_dicts[0], param_dicts[1])
            tls.to_base({'c1':ns1, 'c2':ns2, 'out':nsout}, file=base_name)
            print('s1: ', ns1)
            print('s2: ', ns2)
            print('sout: ', nsout)
    return 0
 
#--------------------------------------------------------------
def multiscan_osc(param_dicts, scans, depth=0, base_name='base2.pkl'):
    
    scan = scans[depth]
    for i in scan['vals']:
        #set parameters-------------
        for j in scan['which']:
            param_dicts[j][scan['p_name']]=i
        #run or scan deeper---------
        if depth < len(scans)-1:  #go deeper
            multiscan_osc(param_dicts, scans, depth=depth+1, base_name=base_name)
        else: #run simultion
            # print current parameters
            print('************* ')
            for pi in scans:
                print(pi['p_name']+' = '+
                     str([param_dicts[di][pi['p_name']] 
                         for di in [0]])  )
            #run simulation
            ns1,nsout = symosc(param_dicts[0])
            tls.to_base({'c1':ns1, 'out':nsout}, file=base_name)
            print('s1: ', ns1)
            print('sout: ', nsout)
    return 0
#============================================================================

def multiscan_with_noise(param_dicts, scans, depth=0, base_name='base2.pkl'):
    
    scan = scans[depth]
    for i in scan['vals']:
        #set parameters-------------
        for j in scan['which']:
            param_dicts[j][scan['p_name']]=i
        #run or scan deeper---------
        if depth < len(scans)-1:  #go deeper
            multiscan_with_noise(param_dicts, scans, depth=depth+1, base_name=base_name)
        else: #run simultion
            # print current parameters
            print('************* ')
            for pi in scans:
                print(pi['p_name']+' = '+
                     str([param_dicts[di][pi['p_name']] 
                         for di in [0,1]])  )
            #run simulation
            ns1,ns2,nsout = symcor_with_noise(param_dicts[0], param_dicts[1])
            tls.to_base({'c1':ns1, 'c2':ns2, 'out':nsout}, file=base_name)
            print('s1: ', ns1)
            print('s2: ', ns2)
            print('sout: ', nsout)
    return 0
    
#============================================================================
