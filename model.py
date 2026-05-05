from prysm import (
    mathops, 
    conf,
)
from prysm.mathops import (
    _np,
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
    gaussian
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

import toml

from astropy.io import (
    fits,
    ascii,
)

from Batoid4LOFT.LAZULI_STOP_mark11 import Lazuli_stop, readBulkMotion, readDeformation

from scipy.interpolate import RegularGridInterpolator

from copy import deepcopy

import os
DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/data/'

class telescope:

    def __init__(self, config, optic_opd_data):

        self.cfg = config
        self.optic_data = optic_opd_data

        # wavelengths to propagate
        wvl0 = config['general']['wvl']
        bandwidth = config['general']['bandwidth']
        n_wvls = config['general']['n_wvls']
        self.wvls = np.linspace(wvl0 - wvl0 * bandwidth / 2, wvl0 + wvl0 * bandwidth / 2, n_wvls)

        # telescope pupil
        x, y = make_xy_grid(shape=self.cfg['general']['npix_pupil'], diameter=self.cfg['optics']['m1']['diam'])
        r, t = cart_to_polar(x, y)
        self.pupil = circle(radius=self.cfg['optics']['m1']['diam'] * 0.98 / 2, r=r)

        # calculate throughput
        self._calc_throughput()
        
        # initialize raytrace model
        self.raytrace = Lazuli_stop()

        # calculate field-dependent raytrace OPDs
        self._calc_opds_field()

        # calculate field-dependent optical surface errors
        self._calc_opds_optics()

        # for M1 bending corrections
        self.opd_m1 = np.zeros_like(self.pupil)

        # setup detector
        self._setup_detector()

    def set_m1_bending(self, bending_opd):
        self.m1_bending_opd = bending_opd

    def add_m1_bending(self, bending_opd):
        self.m1_bending_opd += bending_opd

    def move_optics(self, M1_motion=None, M2_motion=None, M3_motion=None, M4_motion=None):

        self.raytrace = Lazuli_stop(M1_dict=M1_motion, M2_dict=M2_motion, M3_dict=M3_motion, M4_dict=M4_motion, motion_loc=1)
        
        # calculate field-dependent raytrace OPDs
        self._calc_opds_field()

        # calculate field-dependent optical surface errors
        self._calc_opds_optics()

    def reset_optics(self,):
        
        self.raytrace = Lazuli_stop()

        # calculate field-dependent raytrace OPDs
        self._calc_opds_field()

        # calculate field-dependent optical surface errors
        self._calc_opds_optics()

    def set_source_parameters(self, magnitudes, positions):

        dict = {}

        for i, (mag, position) in enumerate(zip(magnitudes, positions)):
            dict.update({'star' + str(i + 1) : {'magnitude'  : mag,
                                                'position_x' : position[0],
                                                'position_y' : position[1]}})
            
        self.cfg['sources'] = dict

        # calculate field-dependent raytrace OPDs
        self._calc_opds_field()

        # calculate field-dependent optical surface errors
        self._calc_opds_optics()

    def set_cam_exposure_time(self, exposure_time):
        self.cfg['detector']['t_exp'] = exposure_time

    def set_cam_gain(self, gain):

        self.cfg['detector']['gain'] = gain

        # get [e-/ADU]/read noise/well depth for gain setting
        file = self.cfg['detector']['gain_curve']
        gain_data = ascii.read(DATA_PATH + file)
        gain_interp = interpolate.PchipInterpolator(gain_data.columns[0].data, gain_data.columns[1].data)
        self.e_per_adu = gain_interp(gain)

        file = self.cfg['detector']['read_noise']
        read_data = ascii.read(DATA_PATH + file)
        read_interp = interpolate.PchipInterpolator(read_data.columns[0].data, read_data.columns[1].data)
        self.read_noise = read_interp(gain)

        file = self.cfg['detector']['well_depth']
        well_data = ascii.read(DATA_PATH + file)
        well_interp = interpolate.PchipInterpolator(well_data.columns[0].data, well_data.columns[1].data)
        self.well_depth = well_interp(gain)

    def _create_wavefront(self, wvl, src_magnitude, debug=False):

        dx_pupil = self.cfg['optics']['m1']['diam'] / self.cfg['general']['npix_pupil']
        
        # initialize wavefront
        wavefront = WF.from_amp_and_phase(self.pupil, None, wvl, dx_pupil)

        # calculate wavefront power
        # using vega flux zero point of 702e10 photons/cm^2/s/m from:
        # https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
        dx_sq = (dx_pupil / 10) ** 2                                                # [mm -> cm^2]
        collecting_area = dx_sq * np.sum(self.pupil)                                # [cm^2]
        bandwidth = (self.wvls[-1] - self.wvls[0]) * 1e-6                           # [um -> m]
        flux = 702e10 * collecting_area * bandwidth * 10 ** (-src_magnitude / 2.5)  # [photons/s]

        if debug:
            print(flux)

        # scale wavefront
        wavefront *= np.sqrt(flux / np.sum(np.abs(wavefront.data) ** 2)) / len(self.wvls)

        return wavefront


    def _fwd(self, wvl, src_magnitude, opd_field, opd_optics, debug=False):

        dx_pupil = self.cfg['optics']['m1']['diam'] / self.cfg['general']['npix_pupil']
        wvl0 = self.cfg['general']['wvl']
        
        # initialize wavefront
        incident = self._create_wavefront(wvl, src_magnitude, debug=debug)
        wfs = [incident]

        # apply M1 bending
        m1 = WF.from_amp_and_phase(amplitude=self.pupil, phase=self.opd_m1, wavelength=wvl, dx=dx_pupil)
        post_bending = incident * m1
        wfs.append(post_bending)

        # apply OPD from optics
        optics = WF.from_amp_and_phase(amplitude=self.pupil, phase=opd_optics, wavelength=wvl, dx=dx_pupil)
        post_optics = post_bending * optics
        wfs.append(post_optics)

        # apply field aberration from raytrace
        field = WF.from_amp_and_phase(amplitude=self.pupil, phase=opd_field, wavelength=wvl, dx=dx_pupil)
        post_field = post_optics * field
        wfs.append(post_field)
        
        # fraunhofer prop to detector
        at_detector = post_field.focus_fixed_sampling(efl=self.cfg['general']['f_eff'], dx=self.cfg['detector']['dx_focal'], 
                                                      samples=self.cfg['general']['npix_focal'], shift=(0, 0), method='mdft')
        wfs.append(at_detector)

        if debug:  
            return(wfs)
        
        else:
            return np.abs(at_detector.data) ** 2
        
    def get_dark(self, stacked_frames=1):

        dim = self.cfg['general']['npix_focal']

        dark = np.ones((dim, dim)) * self.dark_current * self.cfg['detector']['t_exp']

        electrons =  dark.ravel()
        read_noise = np.random.normal(0, self.read_noise, (stacked_frames, electrons.size))

        scaling = 1 / self.e_per_adu
        adc_in = electrons + read_noise
        adc_in[adc_in > self.well_depth] = self.well_depth
        adc_out = adc_in * scaling

        adc_max = 2 ** 16 # 16 bit read out
        adc_out[adc_out > adc_max] = adc_max
        adc_out[adc_out < 0] = 0
        adc_out = np.round(adc_out * (adc_max / (self.well_depth * scaling)))

        output = adc_out.reshape((stacked_frames, *dark.shape))
        if stacked_frames == 1:
            output = output[0, :, :]
        else:
             output = np.mean(output, axis=0)

        return output

    def snap(self, stacked_frames=1, debug=False):

        src_magnitudes = [self.cfg['sources'][source]['magnitude'] for source in self.cfg['sources']]
        
        # initialize images
        images = []
        fluxes = []
        
        # loop through sources
        for magnitude, opd_field, opd_optics in zip(src_magnitudes, self.opds_field, self.opds_optics):
            
            # initialize detector fluxes
            detector_fluxes = []

            # loop through wavelengths
            for wvl in self.wvls.tolist():
                
                # use forward model to get detector flux
                detector_flux = self._fwd(wvl, magnitude, opd_field, opd_optics)
                detector_fluxes.append(detector_flux)

            # get psf intensity by summing across fluxes while applying throughput/exposure time
            psf_flux = sum_of_2d_modes(np.array(detector_fluxes), self.throughput)

            # convolve jitter kernel
            jitter_fft = fft.fft2(fft.ifftshift(self.jitter_kernel))
            psf_fft = fft.fft2(fft.ifftshift(psf_flux))
            psf_flux_with_jitter = fft.fftshift(fft.ifft2(jitter_fft * psf_fft)).real
            fluxes.append(psf_flux_with_jitter)

            electrons = psf_flux_with_jitter * self.cfg['detector']['t_exp']
            dark = self.dark_current * self.cfg['detector']['t_exp']

            electrons = (electrons + dark).ravel()
            shot_noise = np.random.poisson(electrons, (stacked_frames, electrons.size))
            read_noise = np.random.normal(0, self.read_noise, shot_noise.shape)

            scaling = 1 / self.e_per_adu
            adc_in = shot_noise + read_noise
            adc_in[adc_in > self.well_depth] = self.well_depth
            if adc_in.any() > self.well_depth > 0:
                print('WARNING: SATURATED PSF')
            adc_out = adc_in * scaling

            adc_max = 2 ** 16 # 16 bit read out
            adc_out[adc_out > adc_max] = adc_max
            adc_out[adc_out < 0] = 0
            adc_out = np.round(adc_out * (adc_max / (self.well_depth * scaling)))

            output = adc_out.reshape((stacked_frames, *psf_flux_with_jitter.shape))
            if stacked_frames == 1:
                output = output[0, :, :]
            else:
                output = np.mean(output, axis=0)

            images.append(output)

        if debug:
            return images, fluxes
        else:
            return images
    

    def _calc_throughput(self):
        
        # initialize
        self.throughput = np.ones(len(self.wvls))

        # grab optics from config
        optics = self.cfg['optics']

        # loop through optics
        for optic in optics:
            
            # get path to coating file
            file = optics[optic]['coating']

            # load coating data
            coat_data = ascii.read(DATA_PATH + file)

            # create interpolator
            wvl = coat_data.columns[0].data
            refl = coat_data.columns[1].data
            interp = interpolate.PchipInterpolator(wvl, refl)

            # interp for wavelengths and multiply into throughput
            self.throughput *= interp(self.wvls * 1e3) # [um -> nm] to match coating data units

        # get path to QE curve
        file = self.cfg['detector']['qe']

        # load QE curve
        qe_data = ascii.read(DATA_PATH + file)

        # create interpolator
        wvl = qe_data.columns[0].data
        qe = qe_data.columns[1].data
        interp = interpolate.PchipInterpolator(wvl, qe)

        # interp for wavelengths and multiply into throughput
        self.throughput *= interp(self.wvls * 1e3) # [um -> nm] to match coating data units

        return 
    
    def _calc_opds_field(self):
        
        # pull source positions from config
        src_positions = [(self.cfg['sources'][source]['position_x'], self.cfg['sources'][source]['position_y']) for source in self.cfg['sources']]
        
        # initialize
        self.opds_field = []

        # loop through field positions
        for position in src_positions:
            
            # get ray data from raytrace
            # dividing by 60 [arcmin -> degrees] for raytrace
            ray_data = self.raytrace.get_OPD(fieldX=position[0] / 60, fieldY=position[1] / 60, npx=self.cfg['general']['npix_pupil'])

            # convert to OPD map
            opd = np.array(ray_data['wavefront'].array.data * ~ray_data['wavefront'].array.mask)

            # convert units to nm and add to list
            self.opds_field.append(opd * self.cfg['general']['wvl'] * 1e3)

        return
    
    def _calc_opds_optics(self):

        self.opds_optics = []

        maps = [self.optic_data[optic]['map'] for optic in self.optic_data]
        dx_vals = [self.optic_data[optic]['dx'] for optic in self.optic_data]

        optics = self.cfg['optics']

        # pull source positions from config
        src_positions = [(self.cfg['sources'][source]['position_x'], self.cfg['sources'][source]['position_y']) for source in self.cfg['sources']]

        for position in src_positions:

            opd = 0

            for optic, map, dx in zip(optics, maps, dx_vals):

                d_beam = optics[optic]['beam_size']

                dim = map.shape[0]

                x_in = y_in = np.linspace(-dx * dim / 2, dx * dim / 2, dim) * 1e3 # REMOVE THIS LATER WHEN FIXING OPD GENERATION
                
                interp = interpolate.RegularGridInterpolator((x_in, y_in), np.array(map))

                if optic == 'm2':

                    ray_data = self.raytrace.get_footprint(surface='220mm CA dia. M2', fieldX=position[0] / 60, fieldY=position[1] / 60)

                    cx = np.sum(ray_data.x) / len(ray_data.x) * 1e3 
                    cy = np.sum(ray_data.y) / len(ray_data.y) * 1e3 

                    ray_data = self.raytrace.get_footprint(surface='220mm CA dia. M2', fieldX=0, fieldY=0)

                    cx -= np.sum(ray_data.x) / len(ray_data.x) * 1e3 
                    cy -= np.sum(ray_data.y) / len(ray_data.y) * 1e3 

                elif optic == 'm3':

                    ray_data = self.raytrace.get_footprint(surface='380X220mm CA M3', fieldX=position[0] / 60, fieldY=position[1] / 60)

                    cx = np.sum(ray_data.x) / len(ray_data.x) * 1e3 
                    cy = np.sum(ray_data.y) / len(ray_data.y) * 1e3 

                    ray_data = self.raytrace.get_footprint(surface='380X220mm CA M3', fieldX=0, fieldY=0)

                    cx -= np.sum(ray_data.x) / len(ray_data.x) * 1e3 
                    cy -= np.sum(ray_data.y) / len(ray_data.y) * 1e3 

                else:

                    cx = 0
                    cy = 0

                x = np.linspace(-d_beam / 2 - cx, d_beam / 2 - cx, self.cfg['general']['npix_pupil'])
                y = np.linspace(-d_beam / 2 - cy, d_beam / 2 - cy, self.cfg['general']['npix_pupil'])
                x_out, y_out = np.meshgrid(x, y)

                opd += interp((x_out, y_out))

            self.opds_optics.append(opd)

        return
    
    def _setup_detector(self):

        self.resolution_angular = np.rad2deg((self.cfg['general']['wvl'] * 1e-6) / (self.cfg['optics']['m1']['diam'] * 1e-3)) * 3600 # [arcsec/lamD]
        self.resolution_spatial = self.cfg['general']['wvl'] * 1e-6 * self.cfg['general']['f_number'] # [m/lamD]
        self.platescale = self.resolution_angular / self.resolution_spatial # [arcsec/m]

        self.jitter_pix = self.cfg['general']['jitter_rms'] / self.platescale / (self.cfg['detector']['dx_focal'] * 1e-6)

        # make jitter kernel
        x, y = make_xy_grid(shape=self.cfg['general']['npix_focal'], dx=1)
        self.jitter_kernel = gaussian(sigma=self.jitter_pix, x=x, y=y, center=(0, 0))
        self.jitter_kernel /= np.sum(self.jitter_kernel)

        # get dark current for detector temp
        file = self.cfg['detector']['dark_current']
        dark_data = ascii.read(DATA_PATH + file)
        dark_interp = interpolate.PchipInterpolator(dark_data.columns[0].data, dark_data.columns[1].data)
        self.dark_current = dark_interp(self.cfg['detector']['temp'])

        # get [e-/ADU]/read noise/well depth for gain setting
        file = self.cfg['detector']['gain_curve']
        gain_data = ascii.read(DATA_PATH + file)
        gain_interp = interpolate.PchipInterpolator(gain_data.columns[0].data, gain_data.columns[1].data)
        self.e_per_adu = gain_interp(self.cfg['detector']['gain'])

        file = self.cfg['detector']['read_noise']
        read_data = ascii.read(DATA_PATH + file)
        read_interp = interpolate.PchipInterpolator(read_data.columns[0].data, read_data.columns[1].data)
        self.read_noise = read_interp(self.cfg['detector']['gain'])

        file = self.cfg['detector']['well_depth']
        well_data = ascii.read(DATA_PATH + file)
        well_interp = interpolate.PchipInterpolator(well_data.columns[0].data, well_data.columns[1].data)
        self.well_depth = well_interp(self.cfg['detector']['gain'])

        return