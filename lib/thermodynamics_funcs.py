from parameters import *
import numpy as np

def pv_star(T):
    # formula from Bolton 1980
    T_C = T - 273.15
    return 611.2 * np.exp(17.67 * T_C / (T_C+243.5))

def qv_star_t(p_0, T):
    p_vap = pv_star(T)
    return eps_v * p_vap / (p_0 - (1-eps_v)*p_vap)
