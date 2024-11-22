import numpy as np
from scipy.integrate import solve_ivp
import scipy.constants
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import minimize_scalar

# unit conversion for pressure / energy_density
particle_to_SI = scipy.constants.e * 1e51
SI_to_geometric = scipy.constants.G / np.power(scipy.constants.c, 4.0)
particle_to_geometric = particle_to_SI * SI_to_geometric
c = 299792458.0
G = 6.6743e-11
Msun = 1.988409870698051e30
solar_mass_in_meter = Msun * G / c / c # solar mass in geometric unit


class EOS_with_CSE(object):
    """
    Create and eos object with an array of (n, p, e) as the
    low-density tail. And extend the eos to higher density with
    speed-of-sound interpolation. And with the corresponding
    (m, r, lambda) relation solved.

    Parameters:
        low_density_eos: dict, with numpy arrays of n, p, and e in fm^-3, MeV fm^-3 and MeV fm^-3
        n_connect: float, take the low density eos up to the given number density (default: 0.16)
        n_lim: float, having the eos extend to the given number density (default: 2)
        N_seg: int, number of speed-of-sound extension segments (default: 5)
        cs2_limt: float, speed-of-sound squared limit in c^2 (default: 1)
        seed: int, seed of random draw extension (default: 42)
    """

    def __init__(self, low_density_eos: dict, n_lim=2., seed=42):

        self.seed = seed
        self.n_lim = n_lim

        self.n_array = low_density_eos['n']
        self.p_array = low_density_eos['p']
        self.e_array = low_density_eos['e']
        # self.h_array = low_density_eos['h']
        
        self.__calculate_pseudo_enthalpy()
        self.__construct_all_interpolation()


    def __calculate_pseudo_enthalpy(self):

        intergrand = self.p_array / (self.e_array + self.p_array)
        self.h_array = cumulative_trapezoid(intergrand, np.log(self.p_array), initial=0) + intergrand[0]

    def __construct_all_interpolation(self):

        self.log_energy_density_from_log_pressure = interp1d(np.log(self.p_array),
                                                             np.log(self.e_array), kind='linear',
                                                             fill_value='extrapolate',
                                                             assume_sorted=True)
        self.log_energy_density_from_log_pseudo_enthalpy = interp1d(np.log(self.h_array),
                                                                    np.log(self.e_array), kind='linear',
                                                                    fill_value='extrapolate',
                                                                    assume_sorted=True)
        self.log_energy_density_from_log_number_density = interp1d(np.log(self.n_array),
                                                                   np.log(self.e_array), kind='linear',
                                                                   fill_value='extrapolate',
                                                                   assume_sorted=True)

        self.log_pressure_from_log_energy_density = interp1d(np.log(self.e_array),
                                                             np.log(self.p_array), kind='linear',
                                                             fill_value='extrapolate',
                                                             assume_sorted=True)
        self.log_pressure_from_log_number_density = interp1d(np.log(self.n_array),
                                                             np.log(self.p_array), kind='linear',
                                                             fill_value='extrapolate',
                                                             assume_sorted=True)
        self.log_pressure_from_log_pseudo_enthalpy = interp1d(np.log(self.h_array),
                                                              np.log(self.p_array), kind='linear',
                                                              fill_value='extrapolate',
                                                              assume_sorted=True)

        self.log_number_density_from_log_pressure = interp1d(np.log(self.p_array),
                                                             np.log(self.n_array), kind='linear',
                                                             fill_value='extrapolate',
                                                             assume_sorted=True)
        self.log_number_density_from_log_pseudo_enthalpy = interp1d(np.log(self.h_array),
                                                                    np.log(self.n_array), kind='linear',
                                                                    fill_value='extrapolate',
                                                                    assume_sorted=True)
        self.log_number_density_from_log_energy_density = interp1d(np.log(self.e_array),
                                                                   np.log(self.n_array), kind='linear',
                                                                   fill_value='extrapolate',
                                                                   assume_sorted=True)

        self.log_pseudo_enthalpy_from_log_pressure = interp1d(np.log(self.p_array),
                                                              np.log(self.h_array), kind='linear',
                                                              fill_value='extrapolate',
                                                              assume_sorted=True)
        self.log_pseudo_enthalpy_from_log_energy_density = interp1d(np.log(self.e_array),
                                                                    np.log(self.h_array), kind='linear',
                                                                    fill_value='extrapolate',
                                                                    assume_sorted=True)
        self.log_pseudo_enthalpy_from_log_number_density = interp1d(np.log(self.n_array),
                                                                    np.log(self.h_array), kind='linear',
                                                                    fill_value='extrapolate',
                                                                    assume_sorted=True)

        self.log_dedp_from_log_pressure = interp1d(np.log(self.p_array),
                                                   np.gradient(np.log(self.e_array), np.log(self.p_array)),
                                                   kind='linear',
                                                   fill_value='extrapolate',
                                                   assume_sorted=True)

    def energy_density_from_pressure(self, p):
        return np.exp(self.log_energy_density_from_log_pressure(np.log(p)))

    def energy_density_from_pseudo_enthalpy(self, h):
        return np.exp(self.log_energy_density_from_log_pseudo_enthalpy(np.log(h)))

    def energy_density_from_number_density(self, n):
        return np.exp(self.log_energy_density_from_log_number_density(np.log(n)))

    def pressure_from_energy_density(self, e):
        return np.exp(self.log_pressure_from_log_energy_density(np.log(e)))

    def pressure_from_pseudo_enthalpy(self, h):
        return np.exp(self.log_pressure_from_log_pseudo_enthalpy(np.log(h)))

    def pressure_from_number_density(self, n):
        return np.exp(self.log_pressure_from_log_number_density(np.log(n)))

    def number_density_from_pressure(self, p):
        return np.exp(self.log_number_density_from_log_pressure(np.log(p)))

    def number_density_from_pseudo_enthalpy(self, h):
        return np.exp(self.log_number_density_from_log_pseudo_enthalpy(np.log(h)))

    def number_density_from_energy_density(self, e):
        return np.exp(self.log_number_density_from_log_energy_density(np.log(e)))

    def pseudo_enthalpy_from_pressure(self, p):
        return np.exp(self.log_pseudo_enthalpy_from_log_pressure(np.log(p)))

    def pseudo_enthalpy_from_number_density(self, n):
        return np.exp(self.log_pseudo_enthalpy_from_log_number_density(np.log(n)))

    def pseudo_enthalpy_from_energy_density(self, e):
        return np.exp(self.log_pseudo_enthalpy_from_log_energy_density(np.log(e)))

    def dedp_from_pressure(self, p):
        e = self.energy_density_from_pressure(p)
        return e / p * self.log_dedp_from_log_pressure(np.log(p))

    def construct_family(self, pcs):

        ndat = len(pcs)
        pc_min = 3.5  # arbitary lower bound pc in MeV fm^-3
        pc_max = self.pressure_from_number_density(self.n_lim * 0.999)
        pcs = np.logspace(np.log10(pc_min), np.log10(pc_max), num=ndat)
        

        # Generate the arrays of mass, radius and k2
        ms = []
        rs = []
        ks = []
        logpcs = []

        for i, pc in enumerate(pcs):
            m, r, k2 = TOVSolver(self, pc)
            ms.append(m)
            rs.append(r)
            ks.append(k2)
            logpcs.append(np.log(pc))

            if len(ms) > 1 and ms[-1] < ms[-2]:
                break

        ms = np.array(ms)
        rs = np.array(rs)
        ks = np.array(ks)

        cs = ms / rs
        ms /= solar_mass_in_meter
        rs /= 1e3

        lambdas = 2. / 3. * ks * np.power(cs, -5.)

        return ms, rs, lambdas


def tov_ode(h, y, eos):
    r, m, H, b = y
    e = eos.energy_density_from_pseudo_enthalpy(h) * particle_to_geometric
    p = eos.pressure_from_pseudo_enthalpy(h) * particle_to_geometric
    dedp = e / p * eos.log_dedp_from_log_pressure(np.log(p / particle_to_geometric))

    A = 1.0 / (1.0 - 2.0 * m / r)
    C1 = 2.0 / r + A * (2.0 * m / (r * r) + 4.0 * np.pi * r * (p - e))
    C0 = A * (
        -(2) * (2 + 1) / (r * r)
        + 4.0 * np.pi * (e + p) * dedp
        + 4.0 * np.pi * (5.0 * e + 9.0 * p)
    ) - np.power(2.0 * (m + 4.0 * np.pi * r * r * r * p) / (r * (r - 2.0 * m)), 2.0)

    drdh = -r * (r - 2.0 * m) / (m + 4.0 * np.pi * r * r * r * p)
    dmdh = 4.0 * np.pi * r * r * e * drdh
    dHdh = b * drdh
    dbdh = -(C0 * H + C1 * b) * drdh

    dydt = [drdh, dmdh, dHdh, dbdh]

    return dydt


def calc_k2(R, M, H, b):

    y = R * b / H
    C = M / R

    num = (
        (8.0 / 5.0)
        * np.power(1 - 2 * C, 2.0)
        * np.power(C, 5.0)
        * (2 * C * (y - 1) - y + 2)
    )
    den = (
        2
        * C
        * (
            4 * (y + 1) * np.power(C, 4)
            + (6 * y - 4) * np.power(C, 3)
            + (26 - 22 * y) * C * C
            + 3 * (5 * y - 8) * C
            - 3 * y
            + 6
        )
    )
    den -= (
        3
        * np.power(1 - 2 * C, 2)
        * (2 * C * (y - 1) - y + 2)
        * np.log(1.0 / (1 - 2 * C))
    )

    return num / den


def TOVSolver(eos, pc_pp):

    # central values
    hc = eos.pseudo_enthalpy_from_pressure(pc_pp)
    ec = eos.energy_density_from_pressure(pc_pp) * particle_to_geometric
    pc = pc_pp * particle_to_geometric
    dedp_c = eos.dedp_from_pressure(pc_pp)
    dhdp_c = 1.0 / (ec + pc)
    dedh_c = dedp_c / dhdp_c

    # initial values
    dh = -1e-3 * hc
    h0 = hc + dh
    h1 = -dh
    r0 = np.sqrt(3.0 * (-dh) / 2.0 / np.pi / (ec + 3.0 * pc))
    r0 *= 1.0 - 0.25 * (ec - 3.0 * pc - 0.6 * dedh_c) * (-dh) / (ec + 3.0 * pc)
    m0 = 4.0 * np.pi * ec * np.power(r0, 3.0) / 3.0
    m0 *= 1.0 - 0.6 * dedh_c * (-dh) / ec
    H0 = r0 * r0
    b0 = 2.0 * r0

    y0 = [r0, m0, H0, b0]

    sol = solve_ivp(tov_ode, (h0, h1), y0, args=(eos,), rtol=1e-3, atol=0.0)

    # take one final Euler step to get to the surface
    R = sol.y[0, -1]
    M = sol.y[1, -1]
    H = sol.y[2, -1]
    b = sol.y[3, -1]

    y1 = [R, M, H, b]
    dydt1 = tov_ode(h1, y1, eos)

    for i in range(0, len(y1)):
        y1[i] += dydt1[i] * (0.0 - h1)

    R, M, H, b = y1
    k2 = calc_k2(R, M, H, b)

    return M, R, k2