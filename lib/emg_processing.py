# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 15:56:59 2026

@author: Kevin
"""
import numpy as np
import scipy as sp
import lib.utilities as u

def compute_fft(signal, fs, f_min=60, f_max=95):

    N = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_mag = 2.0 / N * np.abs(fft_vals[:N // 2])
    f = np.fft.fftfreq(N, 1/fs)[:N // 2]

    # frequency range of interest
    mask = (f >= f_min) & (f <= f_max)
    f_range = f[mask]
    fft_range = fft_mag[mask]
    
    fft_vib = u.filt_lowpass(fft_range, 200, fs)
    vib_fs_loc = np.argmax(fft_vib)
    vib_fs = np.round(f_range[vib_fs_loc])
    
    return f_range, fft_vib, vib_fs


def filt_notch(signal, fs, vibration_freq, notch_bw=2):
        
    # iirnotch takes normalized frequency: w0 = f0 / (fs/2)
    w0 = vibration_freq / (fs/2)
    Q = vibration_freq / notch_bw  # quality factor: f0 / BW
    b_notch, a_notch = sp.signal.iirnotch(w0, Q)
    
    # Apply notch filter with zero-phase
    filtered = sp.signal.filtfilt(b_notch, a_notch, signal)

    return filtered


def preprocess(raw_signal, fs, bp_low, bp_high, envelope, notch):
    
    offset = np.mean(raw_signal[10000:20000])
    emg_corr = raw_signal - offset
    emg_corr2 = filt_notch(emg_corr, fs, 100) # powerline harmonic
    emg_bandpass = u.filt_bandpass(emg_corr2, fs, bp_low, bp_high)
    
    if notch == None:     
        emg_rect = np.abs(emg_bandpass)
    else:
        emg_notch = filt_notch(emg_bandpass, fs, notch)
        emg_rect = np.abs(emg_notch)
        
    emg_envelope = u.filt_lowpass(emg_rect, envelope, fs)
        
    
    return emg_bandpass, emg_envelope

