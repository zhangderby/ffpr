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

from astropy.io import (
    fits,
    ascii,
)

from Batoid4LOFT.LAZULI_STOP import Lazuli_stop, readBulkMotion, readDeformation

from scipy.interpolate import RegularGridInterpolator

from copy import deepcopy


class off_axis_3m_TMA:

    def __init__(self, opd_maps, config_stp, config_wcc, data_path_stp, data_path_wcc, use_raytrace=True):

        ##### MODEL SETUP #####
        self.cfg_obs = config_stp['observatory']
        self.cfg_tele = config_stp['telescope']
        self.cfg_wcc = config_wcc['common_params']
        self.cfg_e2es = config_wcc['E2ES_default']
        self.path_stp = data_path_stp
        self.path_wcc = data_path_wcc

        # sampling parameters
        self.npix = int(self.cfg_e2es['simulation']['beam_sampling'])  # beam sampling [pix]
        self.fov = int(self.cfg_e2es['simulation']['fov'])             # FOV on detector [pix]

        # get model wavelengths from reference wavlength/bandwidth
        self.wvl0 = float(self.cfg_wcc['spectrum']['wvl_reference']) * 1e6   # ref wavelength [m -> um]
        bw = float(self.cfg_wcc['spectrum']['bandwidth'])               # bandwidth as a fraction of ref wavelength
        n_wvl = int(self.cfg_e2es['simulation']['wvl_sampling'])        # number of wavelengths to sample the spectrum
        self.wvls = np.linspace(self.wvl0 - self.wvl0 * bw / 2, self.wvl0 + self.wvl0 * bw / 2, n_wvl)

        # throughput calculation
        self.throughput = np.ones((len(self.wvls),))

        for optic in self.cfg_tele['optics']:

            # get path to coating data from config
            path2 = self.cfg_tele['optics'][optic]['coating_refl']
            
            # load reflectivity curve and create interpolator
            coat_data = ascii.read(self.path_stp + path2)
            wvl = coat_data.columns[0].data
            refl = coat_data.columns[1].data
            refl_interp = interpolate.PchipInterpolator(wvl, refl)

            # interp for model wavelengths and multiply into throughput
            self.throughput *= refl_interp(self.wvls * 1e3) # wvls [um -> nm] to match reflectivity curve units        

        # source parameters
        self.src_mags = []
        self.src_pos = []
        for source in self.cfg_wcc['sources']:
            self.src_mags.append(float(self.cfg_wcc['sources'][source]['magnitude'])) # magnitudes [Rmag]
            src_x = float(self.cfg_wcc['sources'][source]['position_x']) / 60         # positions [arcmin -> deg]
            src_y = float(self.cfg_wcc['sources'][source]['position_y']) / 60         # positions [arcmin -> deg]
            self.src_pos.append((src_x, src_y)) 

        # defocus diversity
        self.defocus_vals = np.array(self.cfg_e2es['pr']['defocus_vals'])    # defocus [waves]
        self.defocus_vals *= self.wvl0 * 1e3                                 # [waves] -> [nm]

        # use raytrace to include geometric aberrations?
        self.field_aber = []
        if use_raytrace is True:
            self.raytrace = Lazuli_stop()
            for pos in self.src_pos:
                ray_data = self.raytrace.get_OPD(fieldX=pos[0], fieldY=pos[1], npx=self.npix)
                field_opd = np.array(ray_data['wavefront'].array.data * ~ray_data['wavefront'].array.mask)
                self.field_aber.append(field_opd * self.wvl0 * 1e3) # [waves -> um -> nm]
        else:
            self.raytrace = None
            for pos in self.src_pos:
                self.field_aber.append(np.zeros((self.npix, self.npix)))
        
        # pupil parameters
        D_pupil = float(self.cfg_tele['optics']['m1']['aper_clear_OD']) * 1000   # diameter [m -> mm]
        D_obs = float(self.cfg_tele['optics']['m1']['aper_clear_ID']) * 1000     # M2 obsurcation diameter [m -> mm]
        D_support = float(self.cfg_tele['optics']['m2']['support_width']) * 1000 # M2 support width [m -> mm]
        n_support = int(self.cfg_tele['optics']['m2']['n_supports'])             # number of M2 supports

        # pupil grids
        self.x_pup, self.y_pup = make_xy_grid(self.npix, diameter=D_pupil)
        self.r_pup, self.t_pup = cart_to_polar(self.x_pup, self.y_pup)

        # pupil pixelscale
        self.dx_pup = D_pupil / self.npix               

        # pupil mask
        self.pupil = circle(radius=D_pupil / 2, r=self.r_pup)
        if D_obs > 0:
            self.pupil = self.pupil ^ circle(radius=D_obs / 2, r=self.r_pup)
        if n_support > 0:
            self.pupil = self.pupil & spider(vanes=n_support, width=D_support, x=self.x_pup, y=self.y_pup)

        # defocus map  
        r_pup_norm = self.r_pup / (D_pupil / 2)
        self.defocus_map = hopkins(0, 2, 0, r_pup_norm, self.t_pup, 0) 

        # grab f/# and EFL
        fno = float(self.cfg_tele['general']['f_number'])  # working f/#
        self.efl = fno * D_pupil    # effective focal length [mm]

        # optic OPDs
        self.opds = []

        for optic in self.cfg_tele['optics']:
            
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
            self.opds.append(np.array(opd_interp((x_f, y_f))))

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
        pix_per_as = m_per_as / (self.dx_detector * 1e-6)                       # [pix/arcsec]
        jitter = self.cfg_obs['pointing']['jitter_rms']                         # RMS pointing jitter [arcsec]
        self.jitter = jitter * pix_per_as                                       # RMS pointing jitter [pix]

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
                psf_with_jitter = ndimage.gaussian_filter(psf_intensity, sigma=self.jitter, mode='nearest')

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