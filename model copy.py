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


class telescope:

    def __init__(self, config, optic_opd_data):

        # reference wavelength and sampled wavelengths in bandwidth
        self.wvl0 = config['general']['wvl']
        bandwidth = config['general']['bandwidth']
        n_wvls = config['general']['n_wvls']
        self.wvls = np.linspace(self.wvl0 - self.wvl0 * bandwidth / 2, self.wvl0 + self.wvl0 * bandwidth / 2, n_wvls)

        # sampling parameters
        self.npix_pupil = config['general']['npix_pupil']
        self.npix_focal = config['general']['npix_focal']

        # source parameters
        self.src_magnitudes = [config['sources'][source]['magnitude'] for source in config['sources']]
        self.src_positions = [(config['sources'][source]['position_x'], config['sources'][source]['position_y'])
                               for source in config['sources']]
        
        # optical parameters
        self.d_pupil = config['optics']['m1']['diam']
        self.f_num = config['general']['f_number']
        self.f_eff = config['general']['f_eff']
        self.

        # telescope pupil
        x, y = make_xy_grid(self.npix_pupil, diameter=self.d_pupil)
        r, t = cart_to_polar(x, y)
        self.pupil = circle(radius=self.d_pupil / 2, r=r)

        # calculate throughput
        self._calc_throughput(config)
        
        # initialize raytrace model
        self.raytrace = Lazuli_stop()

        # calculate field-dependent raytrace OPDs
        self._calc_opds_field()

        # calculate field-dependent optical surface errors
        self._calc_opds_optics(config, optic_opd_data)

    def _calc_opds_optics(self, config, optic_opd_data):

        self.opds_optics = []

        maps = [optic_opd_data[optic]['map'] for optic in optic_opd_data]
        dx_vals = [optic_opd_data[optic]['dx'] for optic in optic_opd_data]

        optics = config['optics']

        for position in self.src_positions:

            opd = 0

            for optic, map, dx in zip(optics, maps, dx_vals):
                







    
        


        # optic OPDs
        self.opds = []

        ################

        for pos in self.src_pos:

            opd = 0

            for optic in self.cfg_tele['optics']:

                if (optic == 'm1') or (optic == 'm4'):
                
                    # get beam diameter on optic
                    D_beam = float(self.cfg_e2es['optics'][optic]['beam_size']) # meters
                    
                    # get opd map data
                    opd_data = opd_maps[optic]['map']

                    # get pixelscale and dimensions
                    pixscl = opd_maps[optic]['dx']
                    dim = opd_data.shape[0]

                    # create input grid and interpolator for OPD map
                    x_i = y_i = truenp.linspace(-pixscl * dim / 2, pixscl * dim / 2, dim)
                    opd_interp = RegularGridInterpolator((x_i, y_i), opd_data)

                    # create output grid and interpolate to match beam sampling in pupil
                    x_f = y_f = truenp.linspace(-D_beam / 2, D_beam / 2, self.npix) # meters
                    x_f, y_f = truenp.meshgrid(x_f, y_f, indexing='ij')
                    opd += np.array(opd_interp((x_f, y_f)))
                
                elif optic == ('m2'):

                    # get rays on optic
                    rv_2 = self.raytrace.get_footprint(surface='220mm CA dia. M2', fieldX=pos[0], fieldY=pos[1])

                    # find footprint centroid
                    cx = np.sum(rv_2.x)/len(rv_2.x) 
                    cy = np.sum(rv_2.y)/len(rv_2.y)

                    # get beam diameter on optic
                    D_beam = float(self.cfg_e2es['optics'][optic]['beam_size']) # meters
                    
                    # get opd map data
                    opd_data = opd_maps[optic]['map']

                    # get pixelscale and dimensions
                    pixscl = opd_maps[optic]['dx']
                    dim = opd_data.shape[0]

                    # create input grid and interpolator for OPD map
                    x_i = y_i = truenp.linspace(-pixscl * dim / 2, pixscl * dim / 2, dim)
                    opd_interp = RegularGridInterpolator((x_i, y_i), opd_data)

                    # create output grid and interpolate to match beam sampling in pupil
                    x_f = truenp.linspace(-D_beam / 2 - cx, D_beam / 2 - cx, self.npix) # meters
                    y_f = truenp.linspace(-D_beam / 2 - cy, D_beam / 2 - cy, self.npix) # meters
                    x_f, y_f = truenp.meshgrid(x_f, y_f, indexing='ij')
                    opd += np.array(opd_interp((x_f, y_f)))

                elif optic == ('m3'):

                    # get rays on optic
                    rv_2 = self.raytrace.get_footprint(surface='220mm CA dia. M2', fieldX=pos[0], fieldY=pos[1])

                    # find footprint centroid
                    cx = np.sum(rv_2.x)/len(rv_2.x) 
                    cy = np.sum(rv_2.y)/len(rv_2.y)

                    # get beam diameter on optic
                    D_beam = float(self.cfg_e2es['optics'][optic]['beam_size']) # meters
                    
                    # get opd map data
                    opd_data = opd_maps[optic]['map']

                    # get pixelscale and dimensions
                    pixscl = opd_maps[optic]['dx']
                    dim = opd_data.shape[0]

                    # create input grid and interpolator for OPD map
                    x_i = y_i = truenp.linspace(-pixscl * dim / 2, pixscl * dim / 2, dim)
                    opd_interp = RegularGridInterpolator((x_i, y_i), opd_data)

                    # create output grid and interpolate to match beam sampling in pupil
                    x_f = truenp.linspace(-D_beam / 2 - cx, D_beam / 2 - cx, self.npix) # meters
                    y_f = truenp.linspace(-D_beam / 2 - cy, D_beam / 2 - cy, self.npix) # meters
                    x_f, y_f = truenp.meshgrid(x_f, y_f, indexing='ij')
                    opd += np.array(opd_interp((x_f, y_f)))



        #####################

        # for optic in self.cfg_tele['optics']:
            
        #     # get beam diameter on optic
        #     D_beam = float(self.cfg_e2es['optics'][optic]['beam_size']) # meters
            
        #     # get opd map data
        #     opd_data = opd_maps[optic]['map']

        #     # get pixelscale and dimensions
        #     pixscl = opd_maps[optic]['dx']
        #     dim = opd_data.shape[0]

        #     # create input grid and interpolator for OPD map
        #     x_i = y_i = truenp.linspace(-pixscl * dim / 2, pixscl * dim / 2, dim)
        #     opd_interp = RegularGridInterpolator((x_i, y_i), opd_data)

        #     # create output grid and interpolate to match beam sampling in pupil
        #     x_f = y_f = truenp.linspace(-D_beam / 2, D_beam / 2, self.npix) # meters
        #     x_f, y_f = truenp.meshgrid(x_f, y_f, indexing='ij')
        #     self.opds.append(np.array(opd_interp((x_f, y_f))))

        # M1 bending
        self.m1_bending_opd = np.zeros((self.npix, self.npix))
   
        # detector parameters
        self.exp_time = float(self.cfg_wcc['sensor']['exposure_time'])          # exposure time [s]
        self.dx_detector = float(self.cfg_wcc['sensor']['pixel_size']) * 1e6    # detector pixel pitch [m/pix -> um/pix]
        self.detector_gain = float(self.cfg_wcc['sensor']['gain'])              # detector gain setting
        self.black_lvl = float(self.cfg_wcc['sensor']['black_level'])           # detector black level (ADU offset)
        temp_detector = float(self.cfg_wcc['sensor']['temp_nominal'])           # detector temperature [celsius]

        # observatory resolution and jitter
        self.resolution_as = np.rad2deg((self.wvl0 * 1e-6) / (D_pupil * 1e-3)) * 3600# angular resolution [arcsec/resolution_unit]       
        self.resolution_m = self.wvl0 * 1e-6 * fno                                   # spatial resolution[m/resolution_unit]
        m_per_as = self.resolution_m / self.resolution_as                       # [m/arcsec] 
        self.pix_per_as = m_per_as / (self.dx_detector * 1e-6)                       # [pix/arcsec]
        jitter = self.cfg_obs['pointing']['jitter_rms']                         # RMS pointing jitter [arcsec]
        self.jitter = jitter * self.pix_per_as                                       # RMS pointing jitter [pix]

        # make jitter kernel
        x, y = make_xy_grid(shape=self.fov, dx=1)
        self.jitter_kernel = gaussian(sigma=self.jitter, x=x, y=y, center=(0, 0))
        self.jitter_kernel /= np.sum(self.jitter_kernel)

        # get QE and apply to throughput
        path2 = self.cfg_wcc['sensor']['qe']
        qe_data = ascii.read(self.path_wcc + path2)
        # qe_data = np.array([list(x) for x in qe_data])  
        # qe_interp = interpolate.PchipInterpolator(qe_data[:, 0], qe_data[:, 1])
        qe_interp = interpolate.PchipInterpolator(qe_data.columns[0].data, qe_data.columns[1].data)
        self.throughput *= qe_interp(self.wvls * 1e3) # wvls [um -> nm] to match qe curve units

        # get dark current for detector temp
        path2 = self.cfg_wcc['sensor']['dark_current']
        dark_data = ascii.read(self.path_wcc + path2)
        dark_interp = interpolate.PchipInterpolator(dark_data.columns[0].data, dark_data.columns[1].data)
        self.dark_current = dark_interp(temp_detector)

        # get [e-/ADU]/read noise/well depth for gain setting
        path2 = self.cfg_wcc['sensor']['gain_curve']
        gain_data = ascii.read(self.path_wcc + path2)
        gain_interp = interpolate.PchipInterpolator(gain_data.columns[0].data, gain_data.columns[1].data)
        self.e_per_adu = gain_interp(self.detector_gain)

        path2 = self.cfg_wcc['sensor']['read_noise']
        read_data = ascii.read(self.path_wcc + path2)
        read_interp = interpolate.PchipInterpolator(read_data.columns[0].data, read_data.columns[1].data)
        self.read_noise = read_interp(self.detector_gain)

        path2 = self.cfg_wcc['sensor']['well_depth']
        well_data = ascii.read(self.path_wcc + path2)
        well_interp = interpolate.PchipInterpolator(well_data.columns[0].data, well_data.columns[1].data)
        self.well_depth = well_interp(self.detector_gain)

    def set_m1_bending(self, bending_opd):
        self.m1_bending_opd = bending_opd

    def add_m1_bending(self, bending_opd):
        self.m1_bending_opd += bending_opd

    def move_optics(self, M1_motion=None, M2_motion=None, M3_motion=None, M4_motion=None):

        self.raytrace = Lazuli_stop(M1_dict=M1_motion, M2_dict=M2_motion, M3_dict=M3_motion, M4_dict=M4_motion, motion_loc=1)
        
        self.field_aber = []
        for pos in self.src_pos:
            ray_data = self.raytrace.get_OPD(fieldX=pos[0], fieldY=pos[1], npx=self.npix)
            field_opd = np.array(ray_data['wavefront'].array.data * ~ray_data['wavefront'].array.mask)
            self.field_aber.append(field_opd * self.wvl0 * 1e3) # [waves -> um -> nm]

    def reset_optics(self,):
        
        self.raytrace = Lazuli_stop()

        self.field_aber = []
        for pos in self.src_pos:
            ray_data = self.raytrace.get_OPD(fieldX=pos[0], fieldY=pos[1], npx=self.npix)
            field_opd = np.array(ray_data['wavefront'].array.data * ~ray_data['wavefront'].array.mask)
            self.field_aber.append(field_opd * self.wvl0 * 1e3) # [waves -> um -> nm]

    def set_source_parameters(self, magnitudes, positions, defocus_vals):
        self.src_mags = magnitudes
        self.defocus_vals = np.array(defocus_vals)
        self.defocus_vals *= ((self.wvls[0] + self.wvls[-1]) / 2) * 1e3   

        self.src_pos = []
        for pos in positions:
            src_x = pos[0] / 60
            src_y = pos[1] / 60
            self.src_pos.append((src_x, src_y))

        if self.raytrace is not None:
            self.field_aber = []
            for pos in self.src_pos:
                ray_data = self.raytrace.get_OPD(fieldX=pos[0], fieldY=pos[1], npx=self.npix)
                field_opd = np.array(ray_data['wavefront'].array.data * ~ray_data['wavefront'].array.mask)
                self.field_aber.append(field_opd * self.wvl0 * 1e3) # [waves -> um -> nm]      

    def set_cam_exposure_time(self, exposure_time):
        self.exp_time = exposure_time

    def set_cam_black_level(self, black_level):
        self.black_lvl = black_level

    def set_cam_gain(self, gain):

        self.detector_gain = gain

        path2 = self.cfg_wcc['sensor']['gain_curve']
        gain_data = ascii.read(self.path_wcc + path2)
        gain_interp = interpolate.PchipInterpolator(gain_data.columns[0].data, gain_data.columns[1].data)
        self.e_per_adu = gain_interp(self.detector_gain)

        path2 = self.cfg_wcc['sensor']['read_noise']
        read_data = ascii.read(self.path_wcc + path2)
        read_interp = interpolate.PchipInterpolator(read_data.columns[0].data, read_data.columns[1].data)
        self.read_noise = read_interp(self.detector_gain)

        path2 = self.cfg_wcc['sensor']['well_depth']
        well_data = ascii.read(self.path_wcc + path2)
        well_interp = interpolate.PchipInterpolator(well_data.columns[0].data, well_data.columns[1].data)
        self.well_depth = well_interp(self.detector_gain)

    def get_opds(self,):

        return {'Optics' : deepcopy(self.opds),
                'Bending': deepcopy(self.m1_bending_opd),
                'Field'  : deepcopy(self.field_aber)}
    

    def _create_wavefront(self, wvl, src_magnitude, debug=False):
        
        # initialize wavefront
        wavefront = WF.from_amp_and_phase(self.pupil, None, wvl, self.dx_pup)

        # calculate wavefront power
        # using vega flux zero point of 702e10 photons/cm^2/s/m from:
        # https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
        dx_sq = (self.dx_pup / 10) ** 2                                             # [mm -> cm^2]
        collecting_area = dx_sq * np.sum(self.pupil)                                # [cm^2]
        bandwidth = (self.wvls[-1] - self.wvls[0]) * 1e-6                           # [um -> m]
        flux = 702e10 * collecting_area * bandwidth * 10 ** (-src_magnitude / 2.5)  # [photons/s]

        if debug:
            print(flux)

        # scale wavefront
        wavefront *= np.sqrt(flux / np.sum(np.abs(wavefront.data) ** 2) / len(self.wvls))

        return wavefront


    def _fwd(self, wvl, src_magnitude, field_aber, defocus_val, debug=False):
        
        # initialize wavefront
        pre = self._create_wavefront(wvl, src_magnitude, debug=debug)
        wfs = [pre]

        # loop through optics
        for i, opd in enumerate(self.opds):
            if i != 0:
                pre = post

            # create complex screen to represent the 633e-9 / 3optic
            optic = WF.from_amp_and_phase(amplitude=self.pupil, phase=opd, wavelength=wvl, dx=self.dx_pup)

            # apply optic to the wavefront
            post = pre * optic
            wfs.append(post)

        # apply M1 bending
        m1_bending = WF.from_amp_and_phase(amplitude=self.pupil, phase=self.m1_bending_opd, wavelength=wvl, dx=self.dx_pup)
        post_bending = post * m1_bending
        wfs.append(post_bending)

        # apply detector defocus
        defocus = WF.from_amp_and_phase(amplitude=self.pupil, phase=self.defocus_map * -defocus_val, wavelength=wvl, dx=self.dx_pup)
        post_defocus = post_bending * defocus
        wfs.append(post_defocus)

        # apply field aberration from raytrace
        field_aberration = WF.from_amp_and_phase(amplitude=self.pupil, phase=field_aber, wavelength=wvl, dx=self.dx_pup)
        post_field_aber = post_defocus * field_aberration
        wfs.append(post_field_aber)
        
        # fraunhofer prop to detector
        at_detector = post_field_aber.focus_fixed_sampling(efl=self.efl, dx=self.dx_detector, samples=self.fov, shift=(0, 0), method='mdft')
        wfs.append(at_detector)

        if debug:  
            return(wfs)
        
        else:
            return np.abs(at_detector.data) ** 2

    def snap(self, stacked_frames=1,):
        
        # initialize images
        images = []
        
        # loop through sources
        for src_magnitude, field_opd, val in zip(self.src_mags, self.field_aber, self.defocus_vals):
            
            # initialize detector fluxes
            detector_fluxes = []

            # loop through wavelengths
            for wvl in self.wvls.tolist():
                
                # use forward model to get detector flux
                detector_flux = self._fwd(wvl, src_magnitude, field_opd, val)
                detector_fluxes.append(detector_flux)

            # get psf intensity by summing across fluxes while applying throughput/exposure time
            psf_intensity = sum_of_2d_modes(np.array(detector_fluxes), self.throughput) * self.exp_time

            frames = []

            for _ in range(stacked_frames):
                # add jitter as gaussian blur
                jitter_fft = fft.fft2(fft.ifftshift(self.jitter_kernel))
                psf_fft = fft.fft2(fft.ifftshift(psf_intensity))
                psf_with_jitter = fft.fftshift(fft.ifft2(jitter_fft * psf_fft)).real

                # add photon noise
                psf_with_photon_noise = np.random.poisson(psf_with_jitter)

                # apply gain
                # adu = np.round(np.ones(psf_with_photon_noise.shape) * self.black_lvl + psf_with_photon_noise / self.e_per_adu)
                adu = np.round(np.ones(psf_with_photon_noise.shape) + psf_with_photon_noise / self.e_per_adu)

                # add dark current
                adu += np.ones(adu.shape) * self.dark_current * self.exp_time

                # add read noise
                frame = np.round(adu + np.random.normal(loc=0, scale=self.read_noise, size=adu.shape))

                # cant read out negative numbers
                frame[frame < 0] = 0

                # saturate above well depth
                if np.sum(frame[frame > self.well_depth]) > 0:
                    print("WARNING: SATURATED PSF")
                    frame[frame > self.well_depth] = self.well_depth

                frames.append(frame)

            image = np.mean(np.array(frames), axis=0)

            images.append(image)

        return images
    

    def _calc_throughput(self, config):
        
        # initialize
        self.throughput = np.ones(len(self.wvls))

        # grab optics from config
        optics = config['optics']

        # loop through optics
        for optic in optics:
            
            # get path to coating file
            path = optics[optic]['coating']

            # load coating data
            coat_data = ascii.read(path)

            # create interpolator
            wvl = coat_data.columns[0].data
            refl = coat_data.columns[1].data
            interp = interpolate.PchipInterpolator(wvl, refl)

            # interp for wavelengths and multiply into throughput
            self.throughput *= interp(self.wvls * 1e3) # [um -> nm] to match coating data units

        # get path to QE curve
        path = config['detector']['qe']

        # load QE curve
        qe_data = ascii.read(path)

        # create interpolator
        wvl = qe_data.columns[0].data
        qe = qe_data.columns[1].data
        interp = interpolate.PchipInterpolator(wvl, qe)

        # interp for wavelengths and multiply into throughput
        self.throughput *= interp(self.wvls * 1e3) # [um -> nm] to match coating data units

        return 
    
    def _calc_opds_field(self):
        
        # initialize
        self.opds_field = []

        # loop through field positions
        for position in self.src_positions:
            
            # get ray data from raytrace
            # dividing by 60 [arcmin -> degrees] for raytrace
            ray_data = self.raytrace.get_OPD(fieldX=position[0] / 60, fieldY=position[1] / 60, npx=self.npix_pupil)

            # convert to OPD map
            opd = np.array(ray_data['wavefront'].array.data * ~ray_data['wavefront'].array.mask)

            # convert units to nm and add to list
            self.opds_field.append(opd * self.wvl0 * 1e3)

        return