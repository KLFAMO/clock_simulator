# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:00:15 2018

@author: Piotr
"""

import sym_tools as st
import numpy as np
    
#============================================================================
# main program --------------------------------------------

#set simulation parameters===================================================

s1 = {'Tmeas':100, 'name':'1', 'subinter_points':250,
      'Tservocycle':2, 'servocycles':2, 'Tservo':1,
     'Tcycle': 'c', 'w':0.1, 'Tint': 'c', 't0int':800e-3,
     'DMdur':200, 'DMperiod_factor':3, 'DM_lineshift_Hz':0.01, 'DM_synctimeshift':'syncDM',
     'DMshape':'rect', 'DMduradd':0.00, 'DMperiod':450,
     'I':'c', 'gtype':'uni',
     'prob_noise_amp':0.01}

s2 = s1.copy()
#s2['Tcycle']=1.001
s2['name']=2
#=============================================================================

#st.symcor(s1, s2, plot=1 )
#print(st.calc_param(s1))

p_dicts = [s1,s2]
scans = [
       # {'p_name':'Tcycle', 'vals':[1.001, 1.], 'which':[1]},
        
        #{'p_name':'DM_timeshift', 'vals':np.arange(1,10,0.5), 'which':[0,1]},
        {'p_name':'DMdur', 'vals':np.arange(0.6,10,0.2), 'which':[0,1]},
        #{'p_name':'Tservo', 'vals':np.array([1,4,10]), 'which':[0,1]},
        {'p_name':'DM_synctimeshift', 'vals':np.arange(0,2,0.01), 'which':[0,1]},
        #{'p_name':'w', 'vals':np.array([0.1,0.5,0.9]), 'which':[0,1]}
#        {'p_name':'DMdur', 'vals':np.arange(1,10,2), 'which':[0,1]}

        ]
st.multiscan(p_dicts, scans, depth=0 , base_name='base07a.pkl')

    




