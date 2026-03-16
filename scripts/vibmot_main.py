# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:54:18 2026

@author: Kevin
"""
import config as c
import lib.utilities as u
import lib.emg_processing as emg
import scripts.data_processing as dp

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

subjects = pd.read_excel(c.rawDataDir / "Participants_VIBMOT.xlsx")


subjID = subjects["ID"][8]
subjIDs = subjects["ID"][0:8]

matches = [item for item in c.rawDataDir.iterdir()
           if item.is_dir() and subjID in item.name]

subjDir = matches[0] / "successful"

### Create folders if not already existant ###
subj_process_path = c.processedDataDir / subjID
subj_mvc_path = subj_process_path / "mvc_results"

os.makedirs(subj_process_path, exist_ok=True)
os.makedirs(subj_mvc_path, exist_ok=True)

subjTrials = list(subjDir.glob("*.txt"))

""" --- Rest Processing --- """
rest_path = u.filter_files(subjTrials, "rest")
rest = dp.rest_processing(rest_path)

""" --- MVC Processing --- """
# First time do this, afterwrads, just load it in as below
mvc = dp.mvc_processing(subj_mvc_path, subjTrials, rest, None)

mvc_file = pd.read_pickle(subj_mvc_path / "mvc_data.pkl")
mvc = mvc_file['mvc']


""" ALL FOR TESTING """
test_mvc_paths = u.filter_files(subjTrials, "mvc")
test_mvc = u.load_txt_file(mva_gm_paths[0])

test_trial = u.load_txt_file(ext_30_vib_paths[2])

arduino = test_trial["Arduino"]
on_off = np.where(arduino >= 2.1)[0] # 1.2V for first waiting second before start
on_off[0]
""" END HERE """

""" Pre-allocate dict for all datat of all subjects """
variables = ["BF", "ST", "VL", "GM", "Pos", "Vel", "Torq", "Arduino"]

all_data = {}

for task in c.conditions["task"]:
    for muscle in c.conditions["muscle"]:
        for vib in c.conditions["vibration"]:
            
            condition_name = f"{task}_{muscle}_{vib}"
            
            all_data[condition_name] = {
                var: pd.DataFrame() for var in variables
            }


for subjID in subjIDs:
    
    matches = [item for item in c.rawDataDir.iterdir()
               if item.is_dir() and subjID in item.name]

    subjDir = matches[0] / "successful"

    ### Create folders if not already existant ###
    subj_process_path = c.processedDataDir / subjID
    subj_mvc_path = subj_process_path / "mvc_results"

    os.makedirs(subj_process_path, exist_ok=True)
    os.makedirs(subj_mvc_path, exist_ok=True)

    subjTrials = list(subjDir.glob("*.txt"))

    """ --- Rest Processing --- """
    rest_path = u.filter_files(subjTrials, "rest")
    rest = dp.rest_processing(rest_path)
    
    """ --- MVC load in --- """
    mvc_file = pd.read_pickle(subj_mvc_path / "mvc_data.pkl")
    mvc = mvc_file['mvc']
    
    """ Pre-allocate dicts for subject data """
    
    subject_data = {}
    
    for task in c.conditions["task"]:
        for muscle in c.conditions["muscle"]:
            for vib in c.conditions["vibration"]:
                
                name = f"{task}_{muscle}_{vib}"
                subject_data[name] = {}
    
    
    """ --- Data preprocessing --- """
    flex_30_vib_paths = u.filter_files(subjTrials, "30", "flex", "vib")
    flex_30_non_paths = u.filter_files(subjTrials, "30", "flex", "non")
    ext_30_vib_paths = u.filter_files(subjTrials, "30", "ext", "vib")
    ext_30_non_paths = u.filter_files(subjTrials, "30", "ext", "non")
    
    flex_60_vib_paths = u.filter_files(subjTrials, "60", "flex", "vib")
    flex_60_non_paths = u.filter_files(subjTrials, "60", "flex", "non")
    ext_60_vib_paths = u.filter_files(subjTrials, "60", "ext", "vib")
    ext_60_non_paths = u.filter_files(subjTrials, "60", "ext", "non")
    
    # Paths needed for data anaylsis of each subject
    paths = [flex_30_vib_paths, flex_30_non_paths, 
             ext_30_vib_paths, ext_30_non_paths,
             flex_60_vib_paths, flex_60_non_paths, 
             ext_60_vib_paths, ext_60_non_paths]
    
    for num, path in enumerate(paths): 
        
        # Fot plot title
        filename = os.path.basename(str(path)) 
        stem = os.path.splitext(filename)[0]       
        title = "_".join(stem.split("_")[:-1])   
        
        if "vib" in title:
            condition = "vib"
        elif "non" in title:
            condition = "non"
        
        data, test_test, bf_bp = dp.data_processing(path, condition, mvc, rest)
    
        subject_data[title] = data
        
    for condition_name, variables_dict in subject_data.items():
    
        for var, trials in variables_dict.items():
    
            mean_series = trials.mean(axis=1)
    
            all_data[condition_name][var][subjID] = mean_series
            
            
diff = "60"
task = "ext"
var = "BF"
            
plt.plot(all_data[diff + "_" + task + "_non"][var].mean(axis=1), label="non") 
plt.plot(all_data[diff + "_" + task + "_vib"][var].mean(axis=1), label="vib")
plt.legend(); plt.show()
    

    fig, axs = plt.subplots(4, 2)
    fig.suptitle(title, fontweight='bold')
    
    axs[0, 0].plot(data['Pos'])
    axs[0, 0].set_title("Position")
    axs[1, 0].plot(data['Vel'])
    axs[1, 0].set_title("Velocity")
    axs[2, 0].plot(data['Torq'])
    axs[2, 0].set_title("Torque")
    axs[3, 0].plot(data['Arduino'])
    axs[3, 0].set_title("Arduino")
    
    axs[0, 1].plot(data['BF'])
    axs[0, 1].set_title("BF")
    axs[1, 1].plot(data['ST'])
    axs[1, 1].set_title("ST")
    axs[2, 1].plot(data['VL'])
    axs[2, 1].set_title("VL")
    axs[3, 1].plot(data['GM'])
    axs[3, 1].set_title("GM")
    
    plt.tight_layout()
    plt.show()
    
    
    
    
    """ Test data with full unfiltered data """
    fig, axs = plt.subplots(4, 2)
    fig.suptitle(title, fontweight='bold')
    
    axs[0, 0].plot(test_test['Pos'])
    axs[0, 0].set_title("Position")
    axs[1, 0].plot(test_test['Vel'])
    axs[1, 0].set_title("Velocity")
    axs[2, 0].plot(test_test['Torq'])
    axs[2, 0].set_title("Torque")
    axs[3, 0].plot(test_test['Arduino'])
    axs[3, 0].set_title("Arduino")
    
    axs[0, 1].plot(test_data['BF'])
    axs[0, 1].set_title("BF")
    axs[1, 1].plot(test_data['ST'])
    axs[1, 1].set_title("ST")
    axs[2, 1].plot(test_data['VL'])
    axs[2, 1].set_title("VL")
    axs[3, 1].plot(test_data['GM'])
    axs[3, 1].set_title("GM")
    
    plt.tight_layout()
    plt.show()


# Position could be Posizione or "", then offset corrections and its equal
# Velocity and Torque are better with 10Hz lp filter
# Arduino not filtered at all I think
# Torque should be normalized by MVC, or 1RM? Chose MVC now as EMG is also normalized to that
    
    
""" --- Data Averaging per condition --- """



