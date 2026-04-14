import numpy as truenp
import matplotlib.pyplot as plt

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

import copy
import pickle

def ensure_np(arg):
    if isinstance(arg, truenp.ndarray):
        return arg
    if hasattr(arg, 'get'):
        return arg.get()
    
def save_pickle(fpath, data, quiet=False):
    out = open(str(fpath), 'wb')
    pickle.dump(data, out)
    out.close()
    if not quiet: print('Saved data to: ', str(fpath))

def load_pickle(fpath):
    infile = open(str(fpath),'rb')
    pkl_data = pickle.load(infile)
    infile.close()
    return pkl_data  

def plot_psfs(psfs, fields):

    fig, axs = plt.subplots(ncols=7, nrows=3, figsize=(15, 5), dpi=150)

    v = np.max(np.array(psfs))

    count = 0
    lol = 0
    for i, ax in enumerate(axs.flat):
        if  i == 0 or i == 6:
            ax.axis('off')
        else:
            im = ax.imshow(psfs[count].get(), cmap='magma', norm='log', vmax=v, vmin=1)
            ax.set_xticks([])
            ax.set_yticks([])

            if count in (0, 5, 12):
                ax.set_ylabel(f'{fields[count][1] * 60:.01f}"', fontsize=14)
            
            if count >= 12:
                ax.set_xlabel(f'{fields[count][0] * 60:.01f}"', fontsize=14)

            if count in (1, 3):
                c = "black"
                l = f"Instr {lol:.0f}"
                lol += 1

                ax.text(64, 138, l,
                        color=c, fontsize=14,
                        ha='center', va='top',
                        bbox=dict(facecolor='white', edgecolor=c, alpha=1, pad=0.1, boxstyle='round'))
            
            count += 1


    cb = plt.colorbar(im, ax=axs, fraction=0.14, pad=0.04, label='Detector Counts')
    cb.set_label('Detector Counts', fontsize=14)
    cb.ax.tick_params(labelsize=14)
    fig.set_tight_layout = True

    return

def plot_opds(opds, pupil, fields):

    data = copy.deepcopy(opds)

    rms_vals = []
    avg_vals = []

    for opd in data:
        opd -= np.mean(opd[pupil])
        rms_vals.append(np.sqrt(np.mean(opd[pupil] ** 2)))
        avg_vals.append(np.mean(opd[pupil]))
        opd[~pupil] = np.nan

    v = 3 * np.max(np.array(rms_vals))

    fig, axs = plt.subplots(ncols=7, nrows=3, figsize=(15, 5), dpi=150)

    count = 0
    for i, ax in enumerate(axs.flat):
        if  i == 0 or i == 6:
            ax.axis('off')
        else:
            im = ax.imshow(data[count].get(), cmap='coolwarm', vmax=v, vmin=-v)
            ax.set_xticks([])
            ax.set_yticks([])

            if count in (0, 5, 12):
                ax.set_ylabel(f'{fields[count][1] * 60:.01f}"', fontsize=14)
            
            if count >= 12:
                ax.set_xlabel(f'{fields[count][0] * 60:.01f}"', fontsize=14)

            l = f'{rms_vals[count]:.0f} nm RMS'
            c = "black"

            ax.text(256, 550, l,
                    color=c, fontsize=11,
                    ha='center', va='top',
                    bbox=dict(facecolor='white', edgecolor=c, alpha=1, pad=0.1, boxstyle='round'))
            
            count += 1


    cb = plt.colorbar(im, ax=axs, fraction=0.14, pad=0.04)
    cb.set_label('OPD (nm)', fontsize=14)
    cb.ax.tick_params(labelsize=14)
    fig.set_tight_layout = True

    return

def generate_freqs(
        Nf=2**18+1, 
        f_min=0, 
        f_max=1000,
    ):
    """_summary_

    Parameters
    ----------
    Nf : Number of samples for the frequency range, optional
         must be supplied as a power of 2 plus 1, by default 2**18+1
    f_min : _type_, optional
        _description_, by default 0*u.Hz
    f_max : _type_, optional
        _description_, by default 100*u.Hz

    Returns
    -------
    _type_
        _description_
    """
    if bin(Nf-1).count('1')!=1: 
        raise ValueError('Must supply number of samples to be a power of 2 plus 1. ')
    del_f = (f_max - f_min)/Nf
    freqs = np.arange(f_min, f_max, del_f)
    Nt = 2*(Nf-1)
    del_t = 1/(2*f_max)
    times = np.linspace(0, (Nt-1)*del_t, Nt)
    return freqs, times

def kneePSD(
        freqs, 
        beta, 
        fn, 
        alpha
    ):
    psd = beta/(1+freqs/fn)**alpha
    try:
        psd.decompose()
        return psd
    except:
        return psd

def generate_time_series(
        psd, 
        f_max, 
        rms=None,  
        seed=123,
    ):
    Nf = len(psd)
    Nt = 2*(Nf-1)
    del_t = 1/(2*f_max)
    times = np.linspace(0, (Nt-1)*del_t, Nt)

    P_fft_one_sided = copy.copy(psd)

    N_P = len(P_fft_one_sided)  # Length of PSD
    N = 2*(N_P - 1)

    # Because P includes both DC and Nyquist (N/2+1), P_fft must have 2*(N_P-1) elements
    P_fft_one_sided[0] = P_fft_one_sided[0] * 2
    P_fft_one_sided[-1] = P_fft_one_sided[-1] * 2
    P_fft_new = np.zeros((N,), dtype=complex)
    P_fft_new[0:int(N/2)+1] = P_fft_one_sided
    P_fft_new[int(N/2)+1:] = P_fft_one_sided[-2:0:-1]

    X_new = np.sqrt(P_fft_new)

    # Create random phases for all FFT terms other than DC and Nyquist
    np.random.seed(seed)
    phases = np.random.uniform(0, 2*np.pi, (int(N/2),))

    # Ensure X_new has complex conjugate symmetry
    X_new[1:int(N/2)+1] = X_new[1:int(N/2)+1] * np.exp(2j*phases)
    X_new[int(N/2):] = X_new[int(N/2):] * np.exp(-2j*phases[::-1])
    X_new = X_new * np.sqrt(N) / np.sqrt(2)

    # This is the new time series with a given PSD
    x_new = fft.ifft(X_new)

    if rms is not None: 
        x_new *= rms/np.sqrt(np.mean(np.square(x_new)))
    # print(np.sqrt(np.sum(np.square(x_new.real))), np.sqrt(np.sum(np.square(x_new.imag))))

    return x_new.real