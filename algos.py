import numpy as truenp

from prysm import (
    mathops, 
    conf,
)
from prysm.mathops import (
    np,
    fft,
    interpolate,
    ndimage,
)
from prysm.coordinates import (
    make_xy_grid, 
    cart_to_polar,
)
from prysm.propagation import Wavefront as WF
from prysm.propagation import (
    focus_fixed_sampling,
    focus_fixed_sampling_backprop
)                              
from prysm.thinlens import (
    defocus_to_image_displacement,
    image_displacement_to_defocus,
)
from prysm.geometry import (
    circle,
    spider,
)
from prysm.polynomials import (
    lstsq,
    noll_to_nm,
    zernike_nm,
    zernike_nm_seq,
    hopkins,
    sum_of_2d_modes,
    sum_of_2d_modes_backprop
)

from scipy.optimize import minimize


def ensure_np(arg):
    if isinstance(arg, truenp.ndarray):
        return arg
    if hasattr(arg, 'get'):
        return arg.get()


class FFPR:
    def __init__(self, optlist, psf_positions, field_modes, field_coeff_interps):
        
        # list of individual PSF optimizers
        self.optlist = optlist 

        # psf positions in the field
        self.psf_positions = psf_positions

        # interpolators for Z4 thru Z11 which return coeffs given a field postion
        # units for field position should be consistent with `psf_positions`
        self.field_interps = field_coeff_interps

        # for calculating field-dependent coeff deviations from nominal
        self.Z4_a = 0
        self.Z4_b = 0
        self.Z4_c = 0

        self.Z5_a = 0
        self.Z5_b = 0
        self.Z5_c = 0

        self.Z6_a = 0
        self.Z6_b = 0
        self.Z6_c = 0

        self.Z7_a = 0
        self.Z7_b = 0
        self.Z7_c = 0
        
        self.Z8_a = 0
        self.Z8_b = 0
        self.Z8_c = 0

        self.Z11_a = 0
        self.Z11_b = 0
        self.Z11_c = 0
        
        # for field-dependent optimization
        self.modes_field = np.array(field_modes)
        self.coeffs_field_nom = [[interp(np.array(position)) for interp in self.field_interps] for position in self.psf_positions]

        # for joint optimization
        self.modes_common = optlist[0].modes
        self.coeffs_common = np.zeros(len(self.modes_common))

        self.costs = []


    def _fwd_calc_coeffs_field(self, position, coeffs_nom):

        coeffs_field = np.zeros(len(self.modes_field))

        # Z4 deviation from nominal varies linearly across the field
        coeffs_field[0] = self.Z4_a * position[0] + self.Z4_b * position[1] + self.Z4_c + coeffs_nom[0]

        # Z5 deviation from nominal varies linearly across the field
        coeffs_field[1] = self.Z5_a * position[0] + self.Z5_b * position[1] + self.Z5_c + coeffs_nom[1]

        # Z6 deviation from nominal varies linearly across the field
        coeffs_field[2] = self.Z6_a * position[0] + self.Z6_b * position[1] + self.Z6_c + coeffs_nom[2]

        # Z7 deviation from nominal varies linearly across the field
        coeffs_field[3] = self.Z7_a * position[0] + self.Z7_b * position[1] + self.Z7_c + coeffs_nom[3]

        # Z8 deviation from nominal varies linearly across the field
        coeffs_field[4] = self.Z8_a * position[0] + self.Z8_b * position[1] + self.Z8_c + coeffs_nom[4]

        # Z9 does not deviate from nominal
        coeffs_field[5] = coeffs_nom[5]

        # Z10 does not deviate from nominal
        coeffs_field[6] = coeffs_nom[6]

        # Z11 deviation from nominal varies linearly across the field
        coeffs_field[7] = self.Z11_a * position[0] + self.Z11_b * position[1] + self.Z11_c + coeffs_nom[7]

        return coeffs_field
    
    
    def _rev_calc_coeffs_field(self, position, phasebar):

        xbar_partial = np.zeros(18)

        # Z4
        xbar_partial[0] = phasebar[0] * position[0]
        xbar_partial[1] = phasebar[0] * position[1]
        xbar_partial[2] = phasebar[0]

        # Z5
        xbar_partial[3] = phasebar[1] * position[0]
        xbar_partial[4] = phasebar[1] * position[1]
        xbar_partial[5] = phasebar[1]

        # Z6
        xbar_partial[6] = phasebar[2] * position[0] 
        xbar_partial[7] = phasebar[2] * position[1]
        xbar_partial[8] = phasebar[2]

        # Z7
        xbar_partial[9] = phasebar[3] * position[0]
        xbar_partial[10] = phasebar[3] * position[1]
        xbar_partial[11] = phasebar[3]

        # Z8
        xbar_partial[12] = phasebar[4] * position[0]
        xbar_partial[13] = phasebar[4] * position[1]
        xbar_partial[14] = phasebar[4]

        # Z11
        xbar_partial[15] = phasebar[7] * position[0]
        xbar_partial[16] = phasebar[7] * position[1]
        xbar_partial[17] = phasebar[7]

        return xbar_partial


    def fwd_field(self, x):

        self.E = 0

        self.Z4_a = x[0]
        self.Z4_b = x[1]
        self.Z4_c = x[2]

        self.Z5_a = x[3]
        self.Z5_b = x[4]
        self.Z5_c = x[5]

        self.Z6_a = x[6]
        self.Z6_b = x[7]
        self.Z6_c = x[8]

        self.Z7_a = x[9]
        self.Z7_b = x[10]
        self.Z7_c = x[11]
        
        self.Z8_a = x[12]
        self.Z8_b = x[13]
        self.Z8_c = x[14]

        self.Z11_a = x[15]
        self.Z11_b = x[16]
        self.Z11_c = x[17]

        for opt, position, coeffs_nom in zip(self.optlist, self.psf_positions, self.coeffs_field_nom):

            opt.init_opd = sum_of_2d_modes(self.modes_common, self.coeffs_common)

            opt.modes = self.modes_field
            
            coeffs_field = self._fwd_calc_coeffs_field(position, coeffs_nom)

            self.E += opt.fwd(x=coeffs_field)

        self.costs.append(self.E / len(self.optlist))

        return self.E
    
        
    def rev_field(self):

        self.xbar = np.zeros(18)

        for opt, position in zip(self.optlist, self.psf_positions):

            phasebar = opt.rev()

            self.xbar += self._rev_calc_coeffs_field(position, phasebar)

        return self.xbar

        
    def fg_field(self, x):

        f = self.fwd_field(x)

        g = self.rev_field()

        return ensure_np(f), ensure_np(g)

        
    def fwd_common(self, x):

        self.E = 0

        self.coeffs_common = np.array(x)

        for opt, position, coeffs_nom in zip(self.optlist, self.psf_positions, self.coeffs_field_nom):

            coeffs_field = self._fwd_calc_coeffs_field(position, coeffs_nom)
            
            opt.init_opd = sum_of_2d_modes(self.modes_field, coeffs_field)

            opt.modes = self.modes_common

            self.E += opt.fwd(x)

        self.costs.append(self.E / len(self.optlist))

        return self.E
    

    def rev_common(self):

        self.xbar = np.zeros(len(self.modes_common))

        for opt in self.optlist:

            self.xbar += opt.rev()

        return self.xbar
    
    
    def fg_common(self, x):

        f = self.fwd_common(x)

        g = self.rev_common()

        return ensure_np(f), ensure_np(g)
    
    
    def minimize_field(self, jac=True, method='L-BFGS-B', options={'maxls' : 10, 'ftol' : 1e-20, 'gtol' : 1e-8, 'disp' : 0, 'maxiter' : 100}):

        result = minimize(self.fg_field, x0=truenp.array([self.Z4_a, self.Z4_b, self.Z4_c, 
                                                          self.Z5_a, self.Z5_b, self.Z5_c,
                                                          self.Z6_a, self.Z6_b, self.Z6_c,
                                                          self.Z7_a, self.Z7_b, self.Z7_c,
                                                          self.Z8_a, self.Z8_b, self.Z8_c,
                                                          self.Z11_a, self.Z11_b, self.Z11_c]), jac=jac, method=method, options=options)

        return result
    

    def minimize_common(self, jac=True, method='L-BFGS-B', options={'maxls' : 10, 'ftol' : 1e-20, 'gtol' : 1e-8, 'disp' : 0, 'maxiter' : 100}):

        result = minimize(self.fg_common, x0=ensure_np(self.coeffs_common), jac=jac, method=method, options=options)

        return result