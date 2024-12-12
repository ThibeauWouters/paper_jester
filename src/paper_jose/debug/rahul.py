#Reference: Appendix C of Phys. Rev. D 89, 064003

import numpy as np
from scipy import interpolate
from scipy.integrate import cumulative_trapezoid, solve_ivp
from typing import Callable

CONV_MeV_fm3_to_g_cm3   = 1.78266181e-36 * 1e48
CONV_MeV_fm3_to_dyn_cm2 = 1.78266181e-36 * 2.99792458e8**2 * 1e52

G     = 6.6743     * 10**(-8)      # Newton's gravitational constant in cgs units
c     = 2.99792458 * 10**10        # Speed of light in cgs units
M_sun = 1.476      * 10**(5)       # Mass of the sun in geometrized units


def tovrhs(t: np.array,
           yp: tuple[np.array, np.array, np.array],
           eos1: Callable,
           eos2: Callable,
           eos3: Callable):
    
    """
    Right hand side of TOV equations.
    TODO: type hinting.

    Returns:
        _type_: _description_
    """
  
    r, m, y = yp
    
    eps  = eos1(t)
    p    = eos2(t)
    Ga   = eos3(t)
    
    dr_dh = -r*(r-2*m)/(m + 4* np.pi * r**3 * p )
    
    dm_dh = 4 * np.pi * r**2 * eps * dr_dh
        
    l1 = (r-2*m)*(y+1)*y/(m+4*np.pi*r**3*p) + y
    l2 = (m-4*np.pi*r**3*eps)*y/(m+4*np.pi*r**3*p) + (4*np.pi*r**3*(5*eps+9*p)-6*r)/(m+4*np.pi*r**3*p)
    l3 = 4*np.pi*r**3*(eps+p)**2/(p*Ga*(m+4*np.pi*r**3*p)) - 4*(m+4*np.pi*r**3*p)/(r-2*m)
  
    dy_dh = l1+l2+l3
    
    return [dr_dh,dm_dh,dy_dh]
    
    
def tov_solve(n: np.array,
              press: np.array,
              epsilon: np.array,
              c2s: np.array,
              rtol: float=1.e-6,
              atol: float=1.e-5,
              TOV_limit: bool=False, 
              ndat_TOV: int = 100,
              nmin: float = 1e-6 * 0.16,
              limit_by_cs2: bool=False):
    """
    # TODO: Add documentation.
    
    Args:
        epsilon (np.array): Epsilon
        press (np.array): Pressure
        c2s (np.array): Speed of sound, dimensionless between 0 and 1. 
        rtol (float, optional): _description_. Defaults to 1.e-6.
        atol (float, optional): _description_. Defaults to 1.e-5.
        TOV_limit (bool, optional): _description_. Defaults to True.
        h_c_min (float, optional): The starting value of the central enthalpies grid. Defaults to 0.04 (what Rahul had).
        ndat:_TOV int

    Returns:
        _type_: _description_
    """
    
    # TODO: check if this is breaking the solver or not
    
    # Limit the input arrays to only take the part that has cs2 that changes:
    if limit_by_cs2:
        idx = np.where(np.abs(c2s - c2s[0]) > 0)[0][0]
        n = n[idx:]
        press = press[idx:]
        epsilon = epsilon[idx:]
        c2s = c2s[idx:]
        
    # Limit by nmin:
    idx = np.where(n > nmin)[0][0]
    n = n[idx:]
    press = press[idx:]
    epsilon = epsilon[idx:]
    c2s = c2s[idx:]
    
    epsilon = epsilon  *   CONV_MeV_fm3_to_g_cm3       * G * M_sun**2 / c**2
    press   = press    *   CONV_MeV_fm3_to_dyn_cm2     * G * M_sun**2 / c**4
    
    Gamma   = (epsilon + press)/press * c2s
    
    enthalpy     = cumulative_trapezoid(1/(epsilon+press), press, initial=0)
    enthalpy     = np.sort(enthalpy)
    max_enthalpy = np.amax(enthalpy)
                
    spl1 = interpolate.splrep(enthalpy, np.log(epsilon),k=1) 
    spl2 = interpolate.splrep(enthalpy, np.log(press),k=1) 
    spl3 = interpolate.splrep(enthalpy, Gamma,k=1) 

    def eos1(h):
        """log(epsilon) as a function of enthalpy"""
        return np.exp(interpolate.splev(h, spl1, der=0))
    
    def eos2(h):
        """log(pressure) as a function of enthalpy"""
        return np.exp(interpolate.splev(h, spl2, der=0))

    def eos3(h):
        """Gamma as a function of enthalpy"""
        return interpolate.splev(h, spl3, der=0)
    
    central_enthalpies = np.geomspace(0.001, max_enthalpy, ndat_TOV)
       
    Radius = np.zeros_like(central_enthalpies)
    Mass   = np.zeros_like(central_enthalpies)    
    Lambda = np.zeros_like(central_enthalpies)
    
    for i, h_c in enumerate(central_enthalpies):
    
        r0 = 1.e-3 
        m0 = 4/3 * np.pi * r0**3 * eos1(h_c)
        y0 = 2.0
        
        initial = r0, m0, y0
                 
        sol = solve_ivp(tovrhs, (h_c, 0.0), initial, args=(eos1,eos2,eos3),method='LSODA',rtol=rtol,atol=atol)

        R = sol.y[0, -1]
        M = sol.y[1, -1]
        C = M/R
        Y = sol.y[2, -1]
        
        Xi = 4*C**3*(13-11*Y+C*(3*Y-2)+2*C**2*(1+Y)) + 3*(1-2*C)**2*(2-Y+2*C*(Y-1))*np.log(1-2*C)+2*C*(6-3*Y+3*C*(5*Y-8))
        
        Lambda[i] = 16/(15*Xi) * (1-2*C)**2*(2+2*C*(Y-1)-Y)
        Radius[i] = R * M_sun * 10**(-5)
        Mass[i]   = M
        
        if TOV_limit:
            if Mass[i] < Mass[i-1]:
                Mass = Mass[:i]
                Radius = Radius[:i]
                Lambda = Lambda[:i]
                break

    p_c = eos2(central_enthalpies)/(CONV_MeV_fm3_to_dyn_cm2*G*M_sun**2/c**4)  
    p_c = p_c[:len(Mass)]    
        
    return Radius,Mass,Lambda,p_c