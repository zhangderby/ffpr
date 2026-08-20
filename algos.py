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

from prysm.x.optym.cost import bias_and_gain_invariant_error

from scipy.optimize import minimize

from functools import partial

import copy


def ensure_np(arg):
    if isinstance(arg, truenp.ndarray):
        return arg
    if hasattr(arg, 'get'):
        return arg.get()
    
class mean_squared_error():

    def fwd(I, D):

        return np.sum((I - D) ** 2)
    
    def rev(I, D):

        return 2 * (I - D)
    
class gain_invariant_error():

    def fwd(I, D):
        t1 = np.sum(I * D) ** 2
        t2 = np.sum(D ** 2) 
        t3 = np.sum(I ** 2)
        return 1 - t1 / (t2 * t3)

    def rev(I, D):
        t1 = np.sum(I * D)
        t2 = np.sum(D ** 2)
        t3 = np.sum(I ** 2)
        return 2 * t1 / (t2 * t3 ** 2) * (I * t1 - D * t3)
    
def BING_BONG(opt, modal_cleanup=True, zonal_cleanup=False, 
              max_kicks=5, iter_per_bing_bong=20, iter_per_psf=20, 
              max_ls=20, thresh_conv=1e-1, frac_conv=5e-2):

    # WYSI blessed shige seed WYSI
    # do not change or 727 years of failure will befall your bloodline
    rng = np.random.default_rng(seed=727)

    # bing bong things
    count = 1
    bing = 1.
    bong = 1.
    kicks = 0
    best_f = 1.
    best_coeffs = 0
    best_field_terms = 0

    # BING BONG
    print('INITIALIZE BING BONG')

    while True: 
        print('---------------')

        fg = partial(opt.fg, opt_param='common', opt_weights=None)
        x0 = copy.deepcopy(opt.coeffs.get())
        minimize(fg, x0=x0, jac=True, method='L-BFGS-B',
                options={'maxls' : max_ls, 'ftol' : 1e-20, 'gtol' : 1e-10, 'disp' : 0, 
                         'maxiter' : iter_per_bing_bong})

        bing = float(opt.costs[-1])
        print(f'BING {count:02.0f} | f = {bing:.2e}')

        if (np.abs(bong - bing) <= (frac_conv * bong)) and bing < thresh_conv:
            print(f'CONVERGED: BING WITHIN {frac_conv * 100:2.0f}% OF BONG, f BELOW {thresh_conv:.2e}')
            print('TERMINATING BING BONG')
            break
        
        fg = partial(opt.fg, opt_param='all', opt_weights=None)
        x0 = np.concatenate((copy.deepcopy(opt.field_terms), copy.deepcopy(opt.coeffs)), axis=0).get()
        minimize(fg, x0=x0, jac=True, method='L-BFGS-B', 
                options={'maxls' : max_ls, 'ftol' : 1e-20, 'gtol' : 1e-10, 'disp' : 0, 'maxiter' : iter_per_bing_bong})

        bong = float(opt.costs[-1])
        print(f'BONG {count:02.0f} | f = {bong:.2e}')

        if bong < thresh_conv:
            if (np.abs(bing - bong) <= (frac_conv * bing)):
                print(f'CONVERGED: BONG WITHIN {frac_conv * 100:2.0f}% OF BING, f BELOW {thresh_conv:.2e}')
                print('TERMINATE BING BONG')
                break

        if bong > thresh_conv:
            if (np.abs(bing - bong) <= (frac_conv * bing)):
                if kicks == max_kicks:
                    print('NO MORE KICKING, WE MIGHT BE COOKED CHAT')
                    print('TERMINATE BING BONG')
                    if bong > best_f:
                        opt.coeffs = best_coeffs
                        opt.field_terms = best_field_terms
                    break
                else:
                    print('STUCK ABOVE CONVERGENCE THRESHOLD, KICKING')
                    if bong < best_f:
                        best_coeffs = copy.deepcopy(opt.coeffs)
                        best_field_terms = copy.deepcopy(opt.field_terms)
                        best_f = copy.deepcopy(bong)
                    if kicks == 0:
                        opt.coeffs[:10] *= 0.5
                        opt.coeffs[10:] *= 0
                        opt.coeffs[2] *= -1
                        opt.coeffs[9] *= -1
                        opt.field_terms *= 0
                    else:
                        opt.coeffs[2:10] *= rng.uniform(low=-1, high=1, size=len(opt.coeffs[2:10]))
                        opt.coeffs[10:] *= 0
                        opt.field_terms *= 0
                    kicks += 1

        count += 1

    if modal_cleanup:
        print('---------------')
        print('INDIVIDUAL PSF OPTIMIZATION: MODAL')

        for pdpr in opt.PDPR_list:
            fg = partial(pdpr.fg, opt_param='coeffs', opt_weights=None)
            x0 = pdpr.coeffs.get()

            minimize(fg, x0=x0, jac=True, method='L-BFGS-B', 
                    options={'maxls' : max_ls, 'ftol' : 1e-20, 'gtol' : 1e-10, 'disp' : 0, 
                            'maxiter' : iter_per_psf})

    if zonal_cleanup:
        print('---------------')
        print('INDIVIDUAL PSF OPTIMIZATION: ZONAL')
        for pdpr in opt.PDPR_list:
            fg = partial(pdpr.fg, opt_param='map', opt_weights=None)
            x0 = pdpr.map.ravel().get()

            minimize(fg, x0=x0, jac=True, method='L-BFGS-B', 
                    options={'maxls' : max_ls, 'ftol' : 1e-20, 'gtol' : 1e-10, 'disp' : 0, 
                            'maxiter' : iter_per_psf})

    print('---------------')
    print('DONE')
    print()


class FFPR2():

    def __init__(self, wvls, amp, max_zernike, fields, divs, efl, psfs, pupil_dx, focal_dx, error, masks, jitter_kernel=None):
        
        # psf fields
        self.fields = fields

        # for defining field-linear terms due to misalignments
        # self.field_terms = np.zeros(4)
        self.field_terms = np.zeros(6)

        # create zernike basis 
        x, y = make_xy_grid(shape=amp.shape, diameter=2)
        r_norm, t = cart_to_polar(x, y)
        nms = [noll_to_nm(i) for i in range(2, max_zernike + 1)]
        zernikes = list(zernike_nm_seq(nms, r_norm, t, norm=True))

        # for some reason, how the basis is normalized REALLY matters
        # having everything normalized to unit peak-to-valley works best
        zernikes = [z - np.min(z[amp]) for z in zernikes]
        zernikes = [z / np.max(z[amp]) for z in zernikes]
        # zernikes = [z - 0.5 for z in zernikes]
        self.coeffs = np.zeros(len(zernikes)) 

        self.costs = []

        # initialize individual PDPR classes for each field
        self.PDPR_list = [PDPR(wvls=wvls, amp=amp, modes=zernikes, coeffs=np.zeros(len(zernikes)), map=np.zeros((amp.shape[0], amp.shape[1])),
                               divs=divs[f], efl=efl, psfs=psfs[f], pupil_dx=pupil_dx, focal_dx=focal_dx, error=error, masks=masks[f],
                               jitter_kernel=jitter_kernel) for f in range(len(fields))]
        
    def fg(self, x, opt_param=None, opt_weights=None):

        x = np.array(x)

        # reset f, g
        self.f = 0
        self.g = 0

        # reset gradients
        # self.field_terms_bar = np.zeros(4)
        self.field_terms_bar = np.zeros(6)
        self.coeffs_bar = 0
        self.maps_bar = 0

        # if no optimization parameter set, default to all
        if opt_param is None:
            opt_param = 'all'

        # set the optimization parameter x
        # for now, this should be 'all', 'field', or 'common'
        if opt_param == 'all':
            # self.field_terms = x[:4]
            # self.coeffs = x[4:]
            self.field_terms = x[:6]
            self.coeffs = x[6:]
        if opt_param == 'field':
            self.field_terms = x
        if opt_param == 'common':
            self.coeffs = x

        # apply weights if provided
        if opt_weights is not None:
            x *= opt_weights

        # loop through PDPR classes
        for PDPR, field in zip(self.PDPR_list, self.fields):

            coeffs = copy.deepcopy(self.coeffs)

            coeffs[2] += self.field_terms[0] * field[0] + self.field_terms[1] * field[1] 
            # coeffs[3] += self.field_terms[2] * field[0] + self.field_terms[3] * field[1]
            # coeffs[4] += -self.field_terms[3] * field[0] + self.field_terms[2] * field[1]
            coeffs[3] += self.field_terms[2] * field[0] + self.field_terms[3] * field[1]
            coeffs[4] += self.field_terms[4] * field[0] + self.field_terms[5] * field[1]

            PDPR.fg(x=coeffs, opt_param='coeffs', opt_weights=None)

            # add model error to f
            self.f += PDPR.f / len(self.fields)

            # add model OPD gradients to total OPD gradients
            self.maps_bar += copy.deepcopy(PDPR.map_bar)

            # convert model OPD gradient to model coeff gradient then add to total coeff gradients
            coeffs_bar = copy.deepcopy(PDPR.coeffs_bar)

            self.coeffs_bar += coeffs_bar

            # convert total coeff gradients to slope/constant gradients
            self.field_terms_bar[0] += coeffs_bar[2] * field[0]
            self.field_terms_bar[1] += coeffs_bar[2] * field[1]
            # self.field_terms_bar[2] += coeffs_bar[3] * field[0] + coeffs_bar[4] * field[1]
            # self.field_terms_bar[3] += coeffs_bar[3] * field[1] - coeffs_bar[4] * field[0]
            self.field_terms_bar[2] += coeffs_bar[3] * field[0]
            self.field_terms_bar[3] += coeffs_bar[3] * field[1]
            self.field_terms_bar[4] += coeffs_bar[4] * field[0]
            self.field_terms_bar[5] += coeffs_bar[4] * field[1]


        # grab the correct gradients
        if opt_param == 'all':
            self.g = np.concatenate((self.field_terms_bar, self.coeffs_bar), axis=0)
        elif opt_param == 'field':
            self.g = self.field_terms_bar
        elif opt_param == 'common':
            self.g = self.coeffs_bar

        # apply weights if provided
        if opt_weights is not None:
            self.g *= opt_weights

        # append f to costs
        self.costs.append(float(self.f))

        return self.f.get(), self.g.get()
        


class PDPR():

    def __init__(self, wvls, amp, modes, coeffs, map, divs, efl, psfs, pupil_dx, focal_dx, error, masks, jitter_kernel=None):

        # defining OPD using modal basis for low freqs + point map for high freqs
        self.modes = np.array(modes)
        self.coeffs = np.array(coeffs)
        self.map = map
        self.opd = np.tensordot(self.modes, self.coeffs, axes=(0, 0)) + self.map

        self.costs = []

        # initialize models
        if type(psfs) == list:
            self.models = [model(wvls=wvls, amp=amp, opd=self.opd, div=div, efl=efl, psf=psf, pupil_dx=pupil_dx, focal_dx=focal_dx, error=error, 
                                 mask=mask, jitter_kernel=jitter_kernel) for div, psf, mask in zip(divs, psfs, masks)]
        else:
            self.models = [model(wvls=wvls, amp=amp, opd=self.opd, div=divs, efl=efl, psf=psfs, pupil_dx=pupil_dx, 
                                 focal_dx=focal_dx, error=error, mask=masks, jitter_kernel=jitter_kernel)]
        
        
    def fg(self, x, opt_param=None, opt_weights=None):

        x = np.array(x)

        # reset f, g
        self.f = 0
        self.g = 0

        # reset gradients
        self.map_bar = 0
        self.coeffs_bar = 0 

        # if no optimization parameter is set, default to coeffs
        if opt_param is None:
            opt_param = 'coeffs'

        # set the optimization parameter as x
        # for now, this should either be 'coeffs' or 'map'
        if opt_param == 'coeffs':
            self.coeffs = x
        elif opt_param == 'map':
            self.map = x.reshape(self.map.shape)

        # recalculate OPD
        self.opd = np.tensordot(self.modes, self.coeffs, axes=(0, 0)) + self.map

        # apply weights if provided
        if opt_weights is not None:
            x *= opt_weights

        # loop through models
        for model in self.models:
            
            # send OPD to model
            model.opd = self.opd

            # update the model
            model.update()

            # add model error to f
            self.f += model.E / len(self.models)

            # add model OPD gradients to total OPD gradients
            self.map_bar += model.opd_bar

            # convert to model coeff gradients and add to total coeff gradients
            self.coeffs_bar += np.tensordot(self.modes, model.opd_bar)

        # grab the correct gradients
        if opt_param == 'coeffs':
            self.g = self.coeffs_bar
        elif opt_param == 'map':
            self.g = self.map_bar.ravel()

        # apply weights if provided
        if opt_weights is not None:
            self.g *= opt_weights

        # append f to costs
        self.costs.append(float(self.f))

        return self.f.get(), self.g.get()



class model:

    def __init__(self, wvls, amp, opd, div, efl, psf, pupil_dx, focal_dx, error, mask, jitter_kernel=None):

        # model parameters
        self.wvls = wvls
        self.amp = amp
        self.opd = opd
        self.div = div
        self.efl = efl
        self.psf = psf
        self.pupil_dx = pupil_dx
        self.focal_dx = focal_dx
        self.error = error
        self.mask = mask
        self.jitter_kernel = jitter_kernel

        # initialize model
        self.update()

    def update(self):

        # initialize intermediate products
        gs = []
        Gs = []
        self.I = 0

        # gradients too
        G_bars = []
        g_bars = []
        self.opd_bar = 0

        # loop through wavelengths
        for wvl in self.wvls:
            
            # create pupil-plane complex wavefront
            g = self.amp * np.exp((2j * np.pi / wvl) * (self.opd + self.div) / 1e3)
            gs.append(g)

            # propagate wavefront to focal plane using a matrix DFT
            G = focus_fixed_sampling(wavefunction=g, input_dx=self.pupil_dx, prop_dist=self.efl, wavelength=wvl,
                                     output_dx=self.focal_dx, output_samples=self.psf.shape[0], shift=(0, 0), method='mdft')
            Gs.append(G)

            # convert the focal-plane complex wavefront to intensity
            self.I += np.abs(G) ** 2 / len(self.wvls)

        # apply jitter if given
        if self.jitter_kernel is not None:
            jitter_fft = fft.fft2(fft.ifftshift(self.jitter_kernel))
            I_fft = fft.fft2(fft.ifftshift(self.I))
            self.I = fft.fftshift(fft.ifft2(jitter_fft * I_fft)).real

        # calculate error between the model PSF and the measured PSF
        # then calculate the gradient of the rror
        self.E, I_bar = bias_and_gain_invariant_error(self.I, self.psf, mask=self.mask)

        # jitter gradient if given
        if self.jitter_kernel is not None:
            jitter_fft = fft.fft2(fft.ifftshift(self.jitter_kernel))
            I_bar_fft = fft.fft2(fft.ifftshift(I_bar))
            I_bar = fft.fftshift(fft.ifft2(np.conj(jitter_fft) * I_bar_fft)).real

        # loop through wavelengths, per-wavelength G, and per-wavelength g
        for wvl, G, g in zip(self.wvls, Gs, gs):

            # calculate the focal-plane complex wavefront gradient with respect to the focal-plane intensity gradient
            G_bar = 2 * I_bar * G 
            G_bars.append(G_bar)

            # calculate the pupil-plane complex wavefront gradient with respect to the focal-plane complex wavefront gradient
            # this is just an inverse matrix-DFT
            g_bar = focus_fixed_sampling_backprop(wavefunction=G_bar, input_dx=self.pupil_dx, prop_dist=self.efl, wavelength=wvl,
                                                 output_dx=self.focal_dx, output_samples=self.amp.shape[0], shift=(0, 0), method='mdft')
            g_bars.append(g_bar)

            # calculate the pupil phase gradient with respect to the pupil-plane complex wavefront gradient, if
            # multiple wavelengths are given then we sum the gradients across wavelengths to get the total gradient
            self.opd_bar += (2 * np.pi / wvl) * np.imag(g_bar * np.conj(g)) / 1e3

        return 




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

        self.Z9_a = 0
        self.Z9_b = 0
        self.Z9_c = 0

        self.Z10_a = 0
        self.Z10_b = 0
        self.Z10_c = 0

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
        coeffs_field[5] = self.Z9_a * position[0] + self.Z9_b * position[1] + self.Z9_c +coeffs_nom[5]

        # Z10 does not deviate from nominal
        coeffs_field[6] = self.Z10_a * position[0] + self.Z10_b * position[1] + self.Z10_c + coeffs_nom[6]

        # Z11 deviation from nominal varies linearly across the field
        coeffs_field[7] = self.Z11_a * position[0] + self.Z11_b * position[1] + self.Z11_c + coeffs_nom[7]

        return coeffs_field
    
    
    def _rev_calc_coeffs_field(self, position, phasebar):

        xbar_partial = np.zeros(24)

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

        # Z9
        xbar_partial[15] = phasebar[5] * position[0]
        xbar_partial[16] = phasebar[5] * position[1]
        xbar_partial[17] = phasebar[5]

        # Z10
        xbar_partial[18] = phasebar[6] * position[0]
        xbar_partial[19] = phasebar[6] * position[1]
        xbar_partial[20] = phasebar[6]

        # Z11
        xbar_partial[21] = phasebar[7] * position[0]
        xbar_partial[22] = phasebar[7] * position[1]
        xbar_partial[23] = phasebar[7]

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

        self.xbar = np.zeros(24)

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
                                                          self.Z9_a, self.Z9_b, self.Z9_c,
                                                          self.Z10_a, self.Z10_b, self.Z10_c,
                                                          self.Z11_a, self.Z11_b, self.Z11_c]), jac=jac, method=method, options=options)

        return result
    

    def minimize_common(self, jac=True, method='L-BFGS-B', options={'maxls' : 10, 'ftol' : 1e-20, 'gtol' : 1e-8, 'disp' : 0, 'maxiter' : 100}):

        result = minimize(self.fg_common, x0=ensure_np(self.coeffs_common), jac=jac, method=method, options=options)

        return result