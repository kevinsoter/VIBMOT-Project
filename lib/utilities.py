# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:55:04 2026

@author: Kevin
"""

import scipy as sp
import pandas as pd
import numpy as np
import re

""" Loader Functions """
def filter_files(files, *keywords):
    keywords = [k.lower() for k in keywords]

    return [f for f in files
        if all(k in f.name.lower() for k in keywords)]


def load_txt_file(file_path):
    # Load the file
    df = pd.read_csv(file_path, sep="\t", quotechar='"', engine='python')

    # Clean column names
    def clean_col(col_name):
        col_name = col_name.replace('"', '')           # remove quotes
        col_name = re.sub(r'^\d+\s*', '', col_name)    # remove leading numbers + optional spaces
        col_name = col_name.replace(' ', '_')          # replace spaces with underscores
        return col_name

    df.columns = [clean_col(c) for c in df.columns]
    
    # --- truncate at first incomplete row ---
    first_bad_row = df.isna().any(axis=1).idxmax()
    if df.isna().any(axis=1).any():
        df = df.iloc[:first_bad_row]

    return df





"""Functions"""

def interpolate_linear(y, length_new_y):
    x = np.linspace(0, len(y)-1, len(y))
    new_x = np.linspace(0, len(y) - 1, length_new_y)
    new_y = np.interp(new_x, x, y.squeeze())[np.newaxis].T
    return new_y

def filt_bandpass(signal, fs, lowcut, highcut, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = sp.signal.butter(order, [low, high], btype="band")
    filtered = sp.signal.filtfilt(b, a, signal)

    return filtered

def filt_lowpass(signal, cutoff, frequency):
    #nyquist      = frequency * 0.5                            # Not necessary as fs is specified
    #norm_cutoff       = cutoff / nyquist              # normalize frequency
    b, a    = sp.signal.butter(2, cutoff, 'low', fs=frequency)#sp.signal.butter(2, w, 'low')
    signal_filt   = sp.signal.filtfilt(b, a, signal)
    return signal_filt

def bandstop_butter(signal, fs, center_freq, bandwidth=2.0, order=2):
    nyq = 0.5 * fs

    low = (center_freq - bandwidth / 2) / nyq
    high = (center_freq + bandwidth / 2) / nyq

    b, a = sp.signal.butter(order, [low, high], btype="bandstop")
    return sp.signal.filtfilt(b, a, signal)

def angular_acceleration(angle, freq):
    angle = np.array(angle)
    dt = 1 / freq
    velocity = np.diff(angle) / dt
    acceleration = np.diff(velocity) / dt
    acceleration = np.pad(acceleration, (2, 0), 'constant', constant_values=0)
    return pd.DataFrame({'Angular_Acceleration': acceleration})

def slope(x1, y1, x2, y2): # for residual analysis
  s = (y2-y1)/(x2-x1)
  return s

def distance(marker1, marker2):
    x1 = marker1.X
    y1 = marker1.Y
    x2 = marker2.X
    y2 = marker2.Y
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_stabilization_point(force_curve, window_size=12, threshold=0.25):
    """
    Returns the index where the force_curve stabilizes (low std dev in sliding window).
    """
    for i in range(len(force_curve) - window_size):
        window = force_curve[i:i + window_size]
        if np.std(window) < threshold:
            return i + window_size // 2  # midpoint of stable window

    return None  # No stabilization point found


""" Slider function for MVC torque baseline """
def update_val(fig, hline, val):
    hline.set_ydata([val, val])
    fig.canvas.draw_idle()

# --- Click handler ---
def onclick(axs, slider_flex, slider_ext, baselines, event):
    if event.inaxes == axs[0]:
        baselines["flex"] = slider_flex.val
        print(f"\nFlexion baseline chosen: {baselines['flex']:.3f}")

    elif event.inaxes == axs[1]:
        baselines["ext"] = slider_ext.val
        print(f"\nExtension baseline chosen: {baselines['ext']:.3f}")

    else:
        return



