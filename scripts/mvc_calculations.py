# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 16:26:00 2026

@author: Kevin
"""



def mvc_processing(save_folder, subjTrials, notch):
    ### Importing ###
    import config as c
    import lib.utilities as u
    import lib.emg_processing as emg

    import pickle
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    ### File creation ###
    save_file = os.path.join(save_folder, "mvc_data.pkl")
    
    mvc_flex_paths = u.filter_files(subjTrials, "mvc", "flex")
    mvc_ext_paths = u.filter_files(subjTrials, "mvc", "ext")
    mva_gm_paths = u.filter_files(subjTrials, "mva")
    
    """ --- MVC Processing --- """
    mvc_values = pd.DataFrame()
    
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
        
    ### Save datat in pickle format ###
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
            "GM": float(mvc_values["GM"].iloc[0])
        }
    }
    
    # save to pickle
    with open(save_file, "wb") as f:
        pickle.dump(mvc_data, f)
        
    print(f"Saved EMG data to {save_file}")
    
    return mvc_data["mvc"]