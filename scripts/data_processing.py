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



def rest_processing(rest_path, dominance):
    # Calculating resting lever-arm position and torq for gravity correction
    # Based on last measurement point (same as in spike2 script)
    rest_values = pd.DataFrame()
    
    trial = u.load_txt_file(rest_path[0])
    if "R" in str(dominance): # flipping torque as isokinet was turned around for L
        torq = trial["Coppia"]
        vel = trial["Velocita"]
        pos = trial["Posizion"]
    elif "L" in str(dominance):
        torq = trial["Coppia"] * -1
        vel = trial["Velocita"] * -1
        pos = trial["Posizion"] * -1
    
    rest_pos = pos.iloc[-1] # shank angle relative to vertical
    rest_torq = (torq.iloc[-1] / c.convFact) * -1 # MaxGET (according to HUMAC manual)
    # -1 as extension is negative and rest position will always be measured as extension
    torq_SD = np.std((torq/c.convFact) * -1)
    rest_vel = np.mean(vel)
    vel_SD = np.std(vel)

    rest_values = {
        "pos": rest_pos,
        "torq": rest_torq,
        "torq_SD": torq_SD,
        "vel": rest_vel,
        "vel_SD": vel_SD
        }

    return rest_values


def gravity_corr(raw_torq, raw_pos, rest_pos, rest_torq, task):
        
    torq_conv = raw_torq / c.convFact
    pos_shank = raw_pos - rest_pos
    
    if "flex" in str(task):
        # Resisted = M_measured/conversion_factor + (MaxGET * cos(shank angle))
        # Raw flexion torque is positive; (+) thus increases this
        torq_exert = torq_conv + (rest_torq * np.cos(np.deg2rad(pos_shank)))
    elif "ext" in str(task):
        # Assissted = M_measured/conversion_factor - (MaxGET * cos(ahank angle))
        # Raw extension torque is negative; (+) thus decreases this
        torq_exert = torq_conv + (rest_torq * np.cos(np.deg2rad(pos_shank)))
        
    return torq_exert


def mvc_processing(save_folder, subjTrials, rest_values, dominance, notch):

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
    
    if "R" in str(dominance):
        flex_torq_all   = np.concatenate(flex_torq_all)
        ext_torq_all    = np.concatenate(ext_torq_all)
    elif "L" in str(dominance):
        flex_torq_all   = np.concatenate(flex_torq_all) * -1
        ext_torq_all    = np.concatenate(ext_torq_all) * -1
        
    
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


def data_processing(path_list, condition, mvc, rest_values, dominance, angle_grid, target_length=5001):
    # Path list of one specific condition
    # Condition either "vib" or "non"
    # Original sample fs is 5000 Hz, target length is 1000 values per s
    
    variables       = ["BF", "ST", "VL", "GM", "Pos", "Vel", "Acc", "Torq", "Arduino"]
    data            = {var: pd.DataFrame() for var in variables}
    data_angle      = {var: pd.DataFrame() for var in variables}
    test            = {var: pd.DataFrame() for var in variables} # can be deleted after testing done
    
    discrete_vars = [
        "AngleEnd",
        "TimePosStart",
        "TorqPeak",
        "TimeTorqPeak",
        "RFD_Peak",
        "TimeRFD_Peak",
        "VelPeak",
        "TimeVelPeak",
        "AccPeak",
        "TimeAccPeak",
        "TorqStart",
        "RFD_avg",
        "BF_iEMG",
        "ST_iEMG",
        "VL_iEMG",
        "GM_iEMG",
        "Pos_Area_Total",
        "Torq_Area_Total",
        "Torq_Area_Start",
        "Torq_Area_End",
        "Ham_Corr",
        "Quad-Ham_Corr"
    ]
    data_discrete = {var: pd.DataFrame() for var in discrete_vars}
    
    for num, path in enumerate(path_list):
                
        filename = os.path.basename(str(path)) 
        trial_name = os.path.splitext(filename)[0]        
        
        col_name = f"trial_{num+1}"
        
        trial = u.load_txt_file(path)
        
        #print(f"Working on {col_name}")
        
        # Define variables
        bf = trial["Bic_Fem"]
        st = trial["Sem_Tend"]
        vl = trial["Vast_Lat"]
        gm = trial["Gast_Med"]
        arduino = trial["Arduino"]
        
        
        if "R" in str(dominance):
            pos = trial["Posizion"]
            vel = trial["Velocita"]
            torq = trial["Coppia"]
        elif "L" in str(dominance):
            pos = trial["Posizion"] * -1
            vel = trial["Velocita"] * -1
            torq = trial["Coppia"] * -1
        
        if condition == "non":
            vib_fs = None
        else:
            ### Compute notch filter fs ###
            f_vib, fft_vib, vib_fs = emg.compute_fft(st, c.fs)
            if vib_fs < 75 or vib_fs > 90:
                print(f"Vib fs {vib_fs} Hz in {trial_name}")
            else:
                pass
        # If testing for vib frequency uncomment below
        # plt.figure(); plt.plot(f_vib, fft_vib)
                
        ### Filter EMG ###
        bf_bp, bf_env = emg.preprocess(bf, c.fs, c.bp_low, c.bp_high, c.lpCutoff, vib_fs)
        st_bp, st_env = emg.preprocess(st, c.fs, c.bp_low, c.bp_high, c.lpCutoff, vib_fs)
        vl_bp, vl_env = emg.preprocess(vl, c.fs, c.bp_low, c.bp_high, c.lpCutoff, vib_fs)
        gm_bp, gm_env = emg.preprocess(gm, c.fs, c.bp_low, c.bp_high, c.lpCutoff, vib_fs)

        ### Normalize EMG ###
        bf_bp_norm = bf_bp / mvc["BF"]
        st_bp_norm = st_bp / mvc["ST"]
        vl_bp_norm = vl_bp / mvc["VL"]
        gm_bp_norm = gm_bp / mvc["GM"]
        
        bf_norm = bf_env / mvc["BF"]
        st_norm = st_env / mvc["ST"]
        vl_norm = vl_env / mvc["VL"]
        gm_norm = gm_env / mvc["GM"]
        
        ### Offset correct position and velocity data (250 ms = 1000 samples)
        pos_offs = pos - np.mean(pos[500:1500])
        vel_offs = vel - np.mean(vel[500:1500])
              
        ### Filter dynamometer data ###
        pos_filt = u.filt_lowpass(pos_offs, c.lpCutoff, c.fs)
        vel_filt = u.filt_lowpass(vel_offs, c.lpCutoff, c.fs) 
        
        ### Add angular acceleration data
        acc = np.gradient(vel_filt, 1/c.fs)
        
        idx = np.where(abs(pos_filt) >= 5.0)[0] # (4° of RoM movement just to clearly be above 0°; to search backwards from)
        idx_bSearch = idx[0]
        
        ### Gravity correct and normalize torque to MVC ###
        torq_corr = gravity_corr(torq, pos, rest_values["pos"], rest_values["torq"], path) 
        
        # Everything needed to offset correct torque based on 1250 ms of stable baseline
        torq_preFilt = u.filt_lowpass(torq_corr, c.lpCutoff, c.fs)
        rfd_pre = np.gradient(torq_preFilt, 1/c.fs) 
        
        window = int(0.25 * c.fs)   # 250 ms = 1250 samples      
        rfd_thresh = 0.7      # xMVC/s
        torq_sd_thresh = 0.18   # 3x average torq_filt SD in rest
        
        for i in range(idx_bSearch - window, window, -1):
        
            torq_window = torq_preFilt[i-window:i]
            rfd_window = rfd_pre[i-window:i]
        
            if (np.std(torq_window) < torq_sd_thresh and
                np.mean(np.abs(rfd_window)) < rfd_thresh):
        
                idx_onset = i
                break
            
        if idx_onset is None:
            ValueError(f"No stable baseline found for trial {trial_name}")
            
        torq_offset = np.mean(torq_preFilt[idx_onset-window:idx_onset])
        
        if "flex" in str(path):
            torq_norm = (torq_corr - torq_offset) / mvc["Flex"] # normalize to flex MVC
            torq_offs = torq_corr - torq_offset # only offset corrected but not normalized
        elif "ext" in str(path):
            torq_norm = (torq_corr - torq_offset) / mvc["Ext"] # normalize to ext MVC   
            torq_offs = (torq_corr - torq_offset) * -1 # flipping because normalization with negative extension MVC also flips; only offset corrected but not normalized
   
        torq_filt  = u.filt_lowpass(torq_norm, c.lpCutoff, c.fs) #' !!! NEEDS TO BE TORQ_NORM !!!'
        torq_offs_filt  = u.filt_lowpass(torq_offs, c.lpCutoff, c.fs)
        
        ### Start & finish times ###
        on_off_early = np.where(arduino >= 1.2)[0] # 1.2V for first waiting second before start; 2.1 for second
        on_off_late = np.where(arduino  >= 2.1)[0]
        start_6s = on_off_early[0] - (2 * c.fs)
        start_5s = on_off_late[0]
        finish = on_off_late[-1]
        arduino_length_5s = (finish - start_5s) / c.fs
        
        if arduino_length_5s < 5.0 :
            print(f"{trial_name} is {arduino_length_5s}s of 5s")
        else:
            pass
        
        ### Flip all Extension curves to positive and all Flexion curves to negative 
        # (Pos already with correct sign; Torq only correctin Extension direction)
        # 5s used for time normalized continuous curves and SPM
        pos_cut     = pos_filt[start_5s : finish]
        if "flex" in str(path):
            vel_cut     = vel_filt[start_5s : finish] * -1
            acc_cut     = acc[start_5s : finish] * -1
            torq_cut    = torq_filt[start_5s : finish] * -1
        elif "ext" in str(path):
            vel_cut     = vel_filt[start_5s : finish] * -1
            acc_cut     = acc[start_5s : finish] * -1
            torq_cut    = torq_filt[start_5s : finish]
        # EMGs always positive
        bf_cut      = bf_norm[start_5s : finish]
        st_cut      = st_norm[start_5s : finish]
        vl_cut      = vl_norm[start_5s : finish]
        gm_cut      = gm_norm[start_5s : finish]
        arduino_cut = arduino[start_5s : finish]
        
        # 6s used for discrete measures
        pos_6s     = pos_filt[start_6s : finish]
        if "flex" in str(path):
            vel_6s     = vel_filt[start_6s : finish] * -1
            acc_6s     = acc[start_6s : finish] * -1
            torq_6s    = torq_filt[start_6s : finish] * -1
            torq_offs_6s = torq_offs_filt[start_6s : finish] * -1
        elif "ext" in str(path):
            vel_6s     = vel_filt[start_6s : finish] * -1
            acc_6s     = acc[start_6s : finish] * -1
            torq_6s    = torq_filt[start_6s : finish]
            torq_offs_6s = torq_offs_filt[start_6s : finish]
        bf_6s      = bf_bp_norm[start_6s : finish]
        st_6s      = st_bp_norm[start_6s : finish]
        vl_6s      = vl_bp_norm[start_6s : finish]
        gm_6s      = gm_bp_norm[start_6s : finish]

        ### Dicrete Measures (taken from 6s, so 1s baseline + 5s movement)
        if "flex" in str(path):
            posEnd = pos_6s[-1] # end position in degrees to compare between conditions
            
            idx = np.where(vel_6s <= -1.0)[0] # (vel_filt above 1.0°/s because baseline around 0.2 in trials and SD 0.2 in rest)
            idx_posStart = idx[0]
            t_posStart = idx_posStart / c.fs * 1000 # in ms
            
            idx_velPeak = np.argmin(vel_6s[0:25000])
            t_velPeak = idx_velPeak / c.fs * 1000 # in ms
            velPeak = vel_6s[idx_velPeak]
            
            idx_accPeak = np.argmin(acc_6s[0:idx_velPeak])
            t_accPeak = idx_accPeak / c.fs * 1000 # in ms
            accPeak = acc_6s[idx_accPeak]
            
            idx_torqPeak = np.argmin(torq_6s[0:idx_velPeak])
            t_torqPeak = idx_torqPeak / c.fs * 1000 # in ms
            torqPeak = torq_6s[idx_torqPeak]
            
            rfd = np.diff(torq_6s[0:idx_torqPeak] * c.fs)
            idx_peakRFD = np.argmin(rfd)
            peak_rfd = rfd[idx_peakRFD]
            t_peakRFD = idx_peakRFD / c.fs * 1000  # in ms
            
            rfd = np.diff(torq_offs_6s) * c.fs
            idx = np.where(rfd[:idx_peakRFD] > -15.0)[0]
            idx_2 = np.where(torq_offs_6s[idx] > -15.0)[0]
            if idx_2.size == 0:
                idx_torqStart = 0
            else:
                idx_torqStart = idx_2[-1] + 1
            t_torqStart = idx_torqStart / c.fs * 1000 # in ms
            rfd_avg = (torqPeak - torq_6s[idx_torqStart]) / ((idx_torqPeak - idx_torqStart) / c.fs)
                                    
        elif "ext" in str(path):
            posEnd = pos_6s[-1] # end position in degrees to compare between conditions
            
            idx = np.where(vel_6s >= 1.0)[0] # (vel_filt above 1.0°/s because baseline around 0.2 in trials and SD 0.2 in rest)
            idx_posStart = idx[0]
            t_posStart = idx_posStart / c.fs * 1000 # in ms
            
            idx_velPeak = np.argmax(vel_6s[0:25000])
            t_velPeak = idx_velPeak / c.fs * 1000 # in ms
            velPeak = vel_6s[idx_velPeak]
            
            idx_accPeak = np.argmax(acc_6s[0:idx_velPeak])
            t_accPeak = idx_accPeak / c.fs * 1000 # in ms
            accPeak = acc_6s[idx_accPeak]
            
            idx_torqPeak = np.argmax(torq_6s[0:idx_velPeak])
            t_torqPeak = idx_torqPeak / c.fs * 1000 # in ms
            torqPeak = torq_6s[idx_torqPeak]
            
            rfd = np.diff(torq_6s[0:idx_torqPeak] * c.fs)
            idx_peakRFD = np.argmax(rfd)
            peak_rfd = rfd[idx_peakRFD]
            t_peakRFD = idx_peakRFD / c.fs * 1000  # in ms
            
            rfd = np.diff(torq_offs_6s) * c.fs
            idx = np.where(rfd[:idx_peakRFD] < 15.0)[0]
            idx_2 = np.where(torq_offs_6s[idx] < 15.0)[0]
            if idx_2.size == 0:
                idx_torqStart = 0
            else:
                idx_torqStart = idx_2[-1] + 1
            t_torqStart = idx_torqStart / c.fs * 1000 # in ms
            rfd_avg = (torqPeak - torq_6s[idx_torqStart]) / ((idx_torqPeak - idx_torqStart) / c.fs)
                        
            
        bf_rise = np.trapezoid(np.abs(bf_6s[idx_torqStart:idx_torqPeak]), dx=1/c.fs)
        st_rise = np.trapezoid(np.abs(st_6s[idx_torqStart:idx_torqPeak]), dx=1/c.fs)
        vl_rise = np.trapezoid(np.abs(vl_6s[idx_torqStart:idx_torqPeak]), dx=1/c.fs)
        gm_rise = np.trapezoid(np.abs(gm_6s[idx_torqStart:idx_torqPeak]), dx=1/c.fs)

        pos_area_total = np.trapezoid(pos_6s[-1 - (5 * c.fs) : -1], dx=1/c.fs)
        torq_area_total = np.trapezoid(torq_6s[-1 - (5 * c.fs) : -1], dx=1/c.fs)
        torq_area_start = np.trapezoid(torq_6s[-1 - (5 * c.fs) : idx_velPeak], dx=1/c.fs)
        torq_area_end = np.trapezoid(torq_6s[idx_velPeak : -1], dx=1/c.fs)
        
        hamstring_corr, p = sp.stats.pearsonr(bf_cut, st_cut)
        ham_corr_z = np.arctanh(hamstring_corr)
        
        hamstring_cut = (bf_cut + st_cut) / 2
        quad_hams_corr, p = sp.stats.pearsonr(vl_cut, hamstring_cut)
        quad_ham_corr_z = np.arctanh(quad_hams_corr)


        ### Angle normalization from 0 to 72.5 degrees (70 degrees as goal with 5% margin of error) 
        pos_angle    = u.interpolate_angle_norm(pos_cut, pos_cut, angle_grid)    
        vel_angle   = u.interpolate_angle_norm(pos_cut, vel_cut, angle_grid)
        acc_angle   = u.interpolate_angle_norm(pos_cut, acc_cut, angle_grid)
        torq_angle  = u.interpolate_angle_norm(pos_cut, torq_cut, angle_grid)
        bf_angle    = u.interpolate_angle_norm(pos_cut, bf_cut, angle_grid)
        st_angle    = u.interpolate_angle_norm(pos_cut, st_cut, angle_grid)
        vl_angle    = u.interpolate_angle_norm(pos_cut, vl_cut, angle_grid)
        gm_angle    = u.interpolate_angle_norm(pos_cut, gm_cut, angle_grid)
        
        ### Time normalize data with resampling to start at start signal and end immediately after
        pos_resamp = u.interpolate_linear(pos_cut, target_length)
        vel_resamp = u.interpolate_linear(vel_cut, target_length)
        acc_resamp = u.interpolate_linear(acc_cut, target_length)
        torq_resamp = u.interpolate_linear(torq_cut, target_length)
        bf_resamp = u.interpolate_linear(bf_cut, target_length)
        st_resamp = u.interpolate_linear(st_cut, target_length)
        vl_resamp = u.interpolate_linear(vl_cut, target_length)
        gm_resamp = u.interpolate_linear(gm_cut, target_length)
        
        arduino_resamp = u.interpolate_linear(arduino_cut, target_length)
        
        ### Allocate to dict
        data["BF"][col_name]        = bf_resamp
        data["ST"][col_name]        = st_resamp
        data["VL"][col_name]        = vl_resamp
        data["GM"][col_name]        = gm_resamp
        data["Pos"][col_name]       = pos_resamp
        data["Vel"][col_name]       = vel_resamp
        data["Acc"][col_name]       = acc_resamp
        data["Torq"][col_name]      = torq_resamp
        data["Arduino"][col_name]   = arduino_resamp
        
        data_angle["BF"][col_name]        = bf_angle
        data_angle["ST"][col_name]        = st_angle
        data_angle["VL"][col_name]        = vl_angle
        data_angle["GM"][col_name]        = gm_angle
        data_angle["Pos"][col_name]       = pos_angle
        data_angle["Vel"][col_name]       = vel_angle
        data_angle["Acc"][col_name]       = acc_angle
        data_angle["Torq"][col_name]      = torq_angle
    
        data_discrete["AngleEnd"][col_name]     = pd.DataFrame({'posEnd': [posEnd]})
        data_discrete["TimePosStart"][col_name] = pd.DataFrame({'t_PosStart': [t_posStart]})
        data_discrete["TorqPeak"][col_name]     = pd.DataFrame({'torqPeak': [torqPeak]})
        data_discrete["TimeTorqPeak"][col_name] = pd.DataFrame({'t_torqPeak':[t_torqPeak]})
        data_discrete["RFD_Peak"][col_name]     = pd.DataFrame({'peak_rfd': [peak_rfd]})
        data_discrete["TimeRFD_Peak"][col_name] = pd.DataFrame({'t_peakRFD': [t_peakRFD]})
        data_discrete["VelPeak"][col_name]      = pd.DataFrame({'velPeak': [velPeak]})
        data_discrete["TimeVelPeak"][col_name]  = pd.DataFrame({'t_velPeak': [t_velPeak]})
        data_discrete["AccPeak"][col_name]      = pd.DataFrame({'accPeak': [accPeak]})
        data_discrete["TimeAccPeak"][col_name]  = pd.DataFrame({'t_accPeak': [t_accPeak]})
        data_discrete["TorqStart"][col_name]    = pd.DataFrame({'t_torqStart': [t_torqStart]})
        data_discrete["RFD_avg"][col_name]      = pd.DataFrame({'rfd_avg': [rfd_avg]})
        data_discrete["BF_iEMG"][col_name]      = pd.DataFrame({'bf_rise': [bf_rise]})
        data_discrete["ST_iEMG"][col_name]      = pd.DataFrame({'st_rise': [st_rise]})
        data_discrete["VL_iEMG"][col_name]      = pd.DataFrame({'vl_rise': [vl_rise]})
        data_discrete["GM_iEMG"][col_name]      = pd.DataFrame({'gm_rise': [gm_rise]})
        data_discrete["Pos_Area_Total"][col_name]   = pd.DataFrame({'pos_area_total': [pos_area_total]})
        data_discrete["Torq_Area_Total"][col_name]  = pd.DataFrame({'torq_area_total': [torq_area_total]})
        data_discrete["Torq_Area_Start"][col_name]  = pd.DataFrame({'torq_area_start': [torq_area_start]})
        data_discrete["Torq_Area_End"][col_name]    = pd.DataFrame({'torq_area_end': [torq_area_end]})
        data_discrete["Ham_Corr"][col_name]         = pd.DataFrame({'ham_corr': [ham_corr_z]})
        data_discrete["Quad-Ham_Corr"][col_name]    = pd.DataFrame({'quad-ham_corr': [quad_ham_corr_z]})

        ### TEST DATA TO CHECK EVERYTHING ###
        pos_test = sp.signal.resample(pos[0 : -1], target_length)
        vel_test = sp.signal.resample(vel_filt[0 : -1], target_length)
        torq_test = sp.signal.resample(torq_offs_filt[0 : -1], target_length)

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
        
    return data, data_angle, data_discrete, test, bf_bp