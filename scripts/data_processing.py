# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 12:16:19 2026

@author: Kevin
"""
import config as c
import lib.utilities as u
import lib.emg_processing as emg

import scipy as sp
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def rest_processing(rest_path):
    # Calculating resting lever-arm position and torq for gravity correction
    # Based on last measurement point (same as in spike2 script)
    rest_values = pd.DataFrame()
    
    trial = u.load_txt_file(rest_path[0])
    pos = trial["Posizion"]
    torq = trial["Coppia"]
    
    rest_pos = pos.iloc[-1] # shank angle relative to vertical
    rest_torq = (torq.iloc[-1] / c.convFact) * -1 # MaxGET (according to HUMAC manual)
    # -1 as extension is negative and rest position will always be measured as extension

    rest_values = {
        "pos": rest_pos,
        "torq": rest_torq
        }

    return rest_values


def gravity_corr(raw_torq, raw_pos, rest_pos, rest_torq, task):
        
    torq_conv = raw_torq / c.convFact
    pos_shank = raw_pos - rest_pos
    
    if "flex" in str(task):
        # Resisted = M_measured/conversion_factor + (MaxGET * cos(shank angle))
        torq_exert = torq_conv + (rest_torq * np.cos(np.deg2rad(pos_shank)))
    elif "ext" in str(task):
        # Assissted = M_measured/conversion_factor - (MaxGET * cos(ahank angle))
        torq_exert = torq_conv + (rest_torq * np.cos(np.deg2rad(pos_shank)))
        
    return torq_exert


def mvc_processing(save_folder, subjTrials, rest_values, notch):

    ### File creation ###
    save_file = os.path.join(save_folder, "mvc_data.pkl")
    
    mvc_flex_paths = u.filter_files(subjTrials, "mvc", "flex")
    mvc_ext_paths = u.filter_files(subjTrials, "mvc", "ext")
    mva_gm_paths = u.filter_files(subjTrials, "mva")
    
    ### Pre-allocate results folder ###
    mvc_values = pd.DataFrame()
    
    """ --- MVC Processing Torques --- """
    flex_torq_all   = []
    ext_torq_all    = []
    
    mvc_paths_total = mvc_flex_paths[0:3] + mvc_ext_paths[0:3]
    
    for path in mvc_paths_total:
        mvc = u.load_txt_file(path)
        torq = mvc["Coppia"].values / c.convFact
        
        torq_filt = u.filt_lowpass(torq, c.lpCutoff, c.fs)
        
        # Sort into correct array
        if "flex" in str(path):
            flex_torq_all.append(torq_filt)
        elif "ext" in str(path):
            ext_torq_all.append(torq_filt)
        
    flex_torq_all   = np.concatenate(flex_torq_all)
    ext_torq_all    = np.concatenate(ext_torq_all)
    
    
    # Pre-allocate baseline values
    baselines = {"flex": None, "ext": None}
    
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10,7))

    axs[0].plot(flex_torq_all, color='blue')
    axs[0].set_title("Flexion torque — adjust slider, then click inside plot to confirm")
    axs[0].set_ylabel("Torque")

    axs[1].plot(ext_torq_all, color='red')
    axs[1].set_title("Extension torque — adjust slider, then click inside plot to confirm")
    axs[1].set_ylabel("Torque")
    axs[1].set_xlabel("Sample index")

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    # --- Horizontal lines ---
    y0_flex = np.min(flex_torq_all)
    y0_ext  = np.max(ext_torq_all)

    hline_flex = axs[0].axhline(y0_flex, color='black', linestyle='--', linewidth=1.5)
    hline_ext  = axs[1].axhline(y0_ext,  color='black', linestyle='--', linewidth=1.5)

    # --- Sliders ---
    slider_ax1 = plt.axes([0.15, 0.06, 0.7, 0.03])
    slider_ax2 = plt.axes([0.15, 0.02, 0.7, 0.03])

    slider_flex = Slider(slider_ax1, "Flex baseline",
                         valmin=flex_torq_all.min(),
                         valmax=flex_torq_all.max(),
                         valinit=y0_flex)

    slider_ext = Slider(slider_ax2, "Ext baseline",
                        valmin=ext_torq_all.min(),
                        valmax=ext_torq_all.max(),
                        valinit=y0_ext)

    # --- Correct callback binding ---
    slider_flex.on_changed(lambda val: u.update_val(fig, hline_flex, val))
    slider_ext.on_changed(lambda val: u.update_val(fig, hline_ext, val))

    # --- Correct click handler binding ---
    _ = fig.canvas.mpl_connect(
        'button_press_event',
        lambda event: u.onclick(axs, slider_flex, slider_ext, baselines, event)
        )

    plt.show(block=True)    
    
    ### Calculate MVC form baseline to peak ###
    # Peak on extension is the minimum / peak on flexion is the maximum
    flex_peak = np.max(flex_torq_all)
    ext_peak = np.min(ext_torq_all)
    
    mvc_flex_peak = flex_peak - baselines['flex']
    mvc_ext_peak = ext_peak - baselines['ext']        
    
    """ --- MVC Processing EMGs --- """
    window_samples = int(c.window_ms / 1000 * c.fs)
    kernel = np.ones(window_samples) / window_samples # For moving average
    
    ### Flexion (BF and ST) ###
    bf_bandpass_all = []
    bf_envelope_all = []
    
    st_bandpass_all = []
    st_envelope_all = []
    
    for path in mvc_flex_paths[0:3]:
        
        mvc_flex = u.load_txt_file(path)
        bf = mvc_flex["Bic_Fem"].values
        st = mvc_flex["Sem_Tend"].values
        bf_bandpass, bf_envelope = emg.preprocess(bf, c.fs, c.bp_low, c.bp_high, c.lpCutoff, notch)
        st_bandpass, st_envelope = emg.preprocess(st, c.fs, c.bp_low, c.bp_high, c.lpCutoff, notch)
        
        bf_bandpass_all.append(bf_bandpass)
        bf_envelope_all.append(bf_envelope)
        st_bandpass_all.append(st_bandpass)
        st_envelope_all.append(st_envelope)
    
    bf_bandpass_all = np.concatenate(bf_bandpass_all)
    bf_envelope_all = np.concatenate(bf_envelope_all)
    st_bandpass_all = np.concatenate(st_bandpass_all)
    st_envelope_all = np.concatenate(st_envelope_all)
    
    bf_env_avg = np.convolve(bf_envelope_all, kernel, mode="valid")
    st_env_avg = np.convolve(st_envelope_all, kernel, mode="valid")
    
    ### Extension (VL) ###
    vl_bandpass_all = []
    vl_envelope_all = []
    
    for path in mvc_ext_paths[0:3]:
        
        mvc_ext = u.load_txt_file(path)
        vl = mvc_ext["Vast_Lat"].values
        vl_bandpass, vl_envelope = emg.preprocess(vl, c.fs, c.bp_low, c.bp_high, c.lpCutoff, notch)
        
        vl_bandpass_all.append(vl_bandpass)
        vl_envelope_all.append(vl_envelope)
    
    vl_bandpass_all = np.concatenate(vl_bandpass_all)
    vl_envelope_all = np.concatenate(vl_envelope_all)
    
    vl_env_avg = np.convolve(vl_envelope_all, kernel, mode="valid")
        
    ### Gastroc Medialis ###
    mva_gm = u.load_txt_file(mva_gm_paths[0])
    gm = mva_gm["Gast_Med"].values
    
    gm_bandpass, gm_envelope = emg.preprocess(gm, c.fs, c.bp_low, c.bp_high, c.lpCutoff, notch)
    
    gm_env_avg = np.convolve(gm_envelope, kernel, mode="valid")
    
    ### Getting actual MVC values ###
    mvc_values.loc[0, "BF"] = np.max(bf_env_avg)
    mvc_values.loc[0, "ST"] = np.max(st_env_avg)
    mvc_values.loc[0, "VL"] = np.max(vl_env_avg)
    mvc_values.loc[0, "GM"] = np.max(gm_env_avg)
    
    ### Plot all MVCs ###
    # Hamstrings
    fig, axs = plt.subplots(2, 1)
    
    # First muscle
    axs[0].plot(bf_bandpass_all)
    axs[0].plot(bf_envelope_all, linewidth=2)
    axs[0].plot(bf_env_avg, linewidth=2)
    axs[0].set_ylabel("BF")
    
    # Second muscle
    axs[1].plot(st_bandpass_all)
    axs[1].plot(st_envelope_all, linewidth=2)
    axs[1].plot(st_env_avg, linewidth=2)
    axs[1].set_ylabel("ST")
    
    fig.suptitle("Hamstring MVC")
    plt.show()
    
    # Quadriceps
    plt.figure()
    plt.plot(vl_bandpass_all, label="Bandpass")
    plt.plot(vl_envelope_all, label="Envelope", linewidth=2)
    plt.plot(vl_env_avg, label="Average", linewidth=2)
    
    plt.title("VL MVC")
    plt.show()
    
    # Gastrocnemius
    plt.figure()
    plt.plot(gm_bandpass, label="Bandpass")
    plt.plot(gm_envelope, label="Envelope", linewidth=2)
    plt.plot(gm_env_avg, label="Average", linewidth=2)
    plt.title("Gastrocnemius")  # last part of the path
    plt.show()
        
    """ --- Save datat in pickle format --- """
    mvc_data = {
        "bandpass": {
            "BF": bf_bandpass_all,
            "ST": st_bandpass_all,
            "VL": vl_bandpass_all,
            "GM": gm_bandpass
        },
        "envelope": {
            "BF": bf_env_avg,
            "ST": st_env_avg,
            "VL": vl_env_avg,
            "GM": gm_env_avg
        },
        "mvc": {
            "BF": float(mvc_values["BF"].iloc[0]),
            "ST": float(mvc_values["ST"].iloc[0]),
            "VL": float(mvc_values["VL"].iloc[0]),
            "GM": float(mvc_values["GM"].iloc[0]),
            "Flex": float(mvc_flex_peak),
            "Ext": float(mvc_ext_peak)
        }
    }
    
    # save to pickle
    with open(save_file, "wb") as f:
        pickle.dump(mvc_data, f)
        
    print(f"Saved EMG data to {save_file}")
    
    return mvc_data["mvc"]


def data_processing(path_list, condition, mvc, rest_values, target_length=25000):
    # Path list of one specific condition
    # Condition either "vib" or "non"
    # Target length is 6s times 5000 Hz
    
    variables = ["BF", "ST", "VL", "GM", "Pos", "Vel", "Torq", "Arduino"]
    data = {var: pd.DataFrame() for var in variables}
    test = {var: pd.DataFrame() for var in variables} # can be deleted after testing done
    
    for num, path in enumerate(path_list):
        
        filename = os.path.basename(str(path)) 
        trial_name = os.path.splitext(filename)[0]        
        
        col_name = f"trial_{num+1}"
        
        trial = u.load_txt_file(path)
        
        # Define variables
        bf = trial["Bic_Fem"]
        st = trial["Sem_Tend"]
        vl = trial["Vast_Lat"]
        gm = trial["Gast_Med"]
        pos = trial["Posizion"]
        vel = trial["Velocita"]
        torq = trial["Coppia"]
        arduino = trial["Arduino"]
        
        if condition == "non":
            vib_fs = None
        else:
            ### Compute notch filter fs ###
            f_vib, fft_vib, vib_fs = emg.compute_fft(st, c.fs)
            if vib_fs < 65 or vib_fs > 95:
                print(f"Vib fs {vib_fs} Hz in {trial_name}")
            else:
                pass
            
        ### Filter EMG ###
        bf_bp, bf_env = emg.preprocess(bf, c.fs, c.bp_low, c.bp_high, c.lpCutoff, vib_fs)
        st_bp, st_env = emg.preprocess(st, c.fs, c.bp_low, c.bp_high, c.lpCutoff, vib_fs)
        vl_bp, vl_env = emg.preprocess(vl, c.fs, c.bp_low, c.bp_high, c.lpCutoff, vib_fs)
        gm_bp, gm_env = emg.preprocess(gm, c.fs, c.bp_low, c.bp_high, c.lpCutoff, vib_fs)

        ### Normalize EMG ###
        bf_norm = bf_env / mvc["BF"]
        st_norm = st_env / mvc["ST"]
        vl_norm = vl_env / mvc["VL"]
        gm_norm = gm_env / mvc["GM"]
        
        ### Gravity correct and normalize torque to MVC ###
        torq_corr = gravity_corr(torq, pos, rest_values["pos"], rest_values["torq"], path)
        if "flex" in str(path):
            torq_norm = torq_corr / mvc["Flex"]
        elif "ext" in str(path):
            torq_norm = torq_corr / mvc["Ext"]
              
        ### Filter dynamometer data ###
        pos_corr = pos - np.mean(pos[1000:3500])
        pos_filt = u.filt_lowpass(pos_corr, c.lpCutoff, c.fs)
        vel_filt = u.filt_lowpass(vel, c.lpCutoff, c.fs)
        torq_filt  = u.filt_lowpass(torq_norm, c.lpCutoff, c.fs)
        
        ### Start & finish times ###
        on_off = np.where(arduino >= 2.1)[0] # 1.2V for first waiting second before start
        start = on_off[0]
        finish = on_off[-1]
        arduino_length = (finish - start) / c.fs
        
        if arduino_length < 5.0 :
            print(f"{trial_name} is {arduino_length}s of 5s")
        else:
            pass
        
        ### Resample data to start 1s before start signal and end immediately after
        pos_cut = sp.signal.resample(pos_filt[start : finish], target_length)
        vel_cut = sp.signal.resample(vel_filt[start : finish], target_length)
        torq_cut = sp.signal.resample(torq_filt[start : finish], target_length)

        bf_cut = sp.signal.resample(bf_norm[start : finish], target_length)
        st_cut = sp.signal.resample(st_norm[start : finish], target_length)
        vl_cut = sp.signal.resample(vl_norm[start : finish], target_length)
        gm_cut = sp.signal.resample(gm_norm[start : finish], target_length)
        
        arduino_cut = sp.signal.resample(arduino[start : finish], target_length)
        
        ### Allocate to dict
        data["BF"][col_name] = bf_cut
        data["ST"][col_name] = st_cut
        data["VL"][col_name] = vl_cut
        data["GM"][col_name] = gm_cut
        data["Pos"][col_name] = pos_cut
        data["Vel"][col_name] = vel_cut
        data["Torq"][col_name] = torq_cut
        data["Arduino"][col_name] = arduino_cut
        
        
        ### TEST DATA TO CHECK EVERYTHING ###
        pos_test = sp.signal.resample(pos[0 : -1], target_length)
        vel_test = sp.signal.resample(vel_filt[0 : -1], target_length)
        torq_test = sp.signal.resample(torq[0 : -1], target_length)

        bf_test = sp.signal.resample(bf_bp[0 : -1], target_length)
        st_test = sp.signal.resample(st_bp[0 : -1], target_length)
        vl_test = sp.signal.resample(vl_norm[0 : -1], target_length)
        gm_test = sp.signal.resample(gm_norm[0 : -1], target_length)
        
        arduino_test = sp.signal.resample(arduino[0 : -1], target_length)
        
        test["BF"][col_name] = bf_test
        test["ST"][col_name] = st_test
        test["VL"][col_name] = vl_test
        test["GM"][col_name] = gm_test
        test["Pos"][col_name] = pos_test
        test["Vel"][col_name] = vel_test
        test["Torq"][col_name] = torq_test
        test["Arduino"][col_name] = arduino_test
        
    return data, test, bf_bp