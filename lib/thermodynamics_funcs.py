from parameters import *
import numpy as np

def pd(p0, qt, qv):
    return p0 * (1.0-qt) / (1.0-qt+eps_vi*qv)

def pv(p0, qt, qv):
    return p0 * eps_vi * qv / (1.0-qt+eps_vi*qv)

def cpm(qt):
    return (1.0-qt)*cpd + qt*cpv

def pv_star(T):
    # formula from Bolton 1980
    T_C = T - 273.15
    return 611.2 * np.exp(17.67 * T_C / (T_C+243.5))

def qv_star_t(p_0, T):
    p_vap = pv_star(T)
    return eps_v * p_vap / (p_0 - (1-eps_v)*p_vap)
