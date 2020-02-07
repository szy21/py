from parameters import *
import thermodynamics_funcs as td
import numpy as np

def sd(pd, T):
    return sd_tilde + cpd*np.log(T/T_tilde) - Rd*np.log(pd/p_tilde)

def sv(pv, T):
    return sv_tilde + cpv*np.log(T/T_tilde) - Rv*np.log(pv/p_tilde)

def sc(L,T):
    return -L/T

def s_tendency(p0, qt, qv, T, qt_tendency, T_tendency):
    pv = td.pv(p0, qt, qv)
    pd = td.pd(p0, qt, qv)
    return td.cpm(qt)*T_tendency/T + (sv(pv, T)-sd(pd, T))*qt_tendency
