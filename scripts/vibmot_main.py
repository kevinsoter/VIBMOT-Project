# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:54:18 2026

@author: Kevin Soter
"""

""" Execute prior to Analysis """
import config as c
import lib.utilities as u
import lib.emg_processing as emg
import scripts.data_processing as dp

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

subjects = pd.read_excel(c.rawDataDir / "Participants_VIBMOT.xlsx")
 

subjID = subjects["ID"][0]
dominance = subjects["Dominance"][subjects.index[subjects['ID'] == subjID][0]]


subjIDs = subjects["ID"][0:30]

matches = [item for item in c.rawDataDir.iterdir()
           if item.is_dir() and subjID in item.name]

subjDir = matches[0] / "successful"
""" END Execution prior to Analysis """

""" Execute for analysis of one single Subject """
### Create folders if not already existant ###
subj_process_path = c.processedDataDir / subjID
subj_mvc_path = subj_process_path / "mvc_results"

os.makedirs(subj_process_path, exist_ok=True)
os.makedirs(subj_mvc_path, exist_ok=True)

subjTrials = list(subjDir.glob("*.txt"))

""" --- Rest Processing --- """
rest_path = u.filter_files(subjTrials, "rest")
rest = dp.rest_processing(rest_path, dominance)

""" --- MVC Processing --- """
try:
    # If file exists it is loaded in
    mvc_file = pd.read_pickle(subj_mvc_path / "mvc_data.pkl")
    mvc = mvc_file['mvc']
except FileNotFoundError:
    # If file does not exist it gets created
    mvc = dp.mvc_processing(subj_mvc_path, subjTrials, rest, dominance, None)
    raise SystemExit

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
    
    #path = ext_60_non_paths # only for testing
    
    # Fot plot title
    filename = os.path.basename(str(path)) 
    stem = os.path.splitext(filename)[0]       
    title = "_".join(stem.split("_")[:-1])   
    
    if "vib" in title:
        condition = "vib"
    elif "non" in title:
        condition = "non"
    
    ### For angle normalization from 0 to 72.5 degrees (70 degrees as goal with 5% margin of error) 
    if "flex" in str(path):
        angle_grid = np.linspace(0, 72.5, 726) * -1
    elif "ext" in str(path):
        angle_grid = np.linspace(0, 72.5, 726)
    
    data, data_angle, data_discrete, test_test, bf_bp = dp.data_processing(path, condition, mvc, rest, dominance, angle_grid)

    plt.figure(); plt.plot(data['Torq'])

    """ Test data with full unfiltered data """
    fig, axs = plt.subplots(4, 2)
    fig.suptitle([title  + "_Time_Norm"], fontweight='bold')

    axs[0, 0].plot(data['Pos'])
    axs[0, 0].set_title("Position")
    axs[1, 0].plot(data['Vel'])
    axs[1, 0].set_title("Velocity")
    axs[2, 0].plot(data["Acc"])
    axs[2, 0].set_title("Acceleration")
    axs[3, 0].plot(data['Torq'])
    axs[3, 0].set_title("Torque")

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
    
    
    fig, axs = plt.subplots(4, 2)
    fig.suptitle([title + "_Angle_Norm"], fontweight='bold')

    axs[0, 0].plot(angle_grid, data_angle['Pos'])
    axs[0, 0].set_title("Position")
    axs[1, 0].plot(angle_grid, data_angle['Vel'])
    axs[1, 0].set_title("Velocity")
    axs[2, 0].plot(angle_grid, data_angle["Acc"])
    axs[2, 0].set_title("Acceleration")
    axs[3, 0].plot(angle_grid, data_angle['Torq'])
    axs[3, 0].set_title("Torque")


    axs[0, 1].plot(angle_grid, data_angle['BF'])
    axs[0, 1].set_title("BF")
    axs[1, 1].plot(angle_grid, data_angle['ST'])
    axs[1, 1].set_title("ST")
    axs[2, 1].plot(angle_grid, data_angle['VL'])
    axs[2, 1].set_title("VL")
    axs[3, 1].plot(angle_grid, data_angle['GM'])
    axs[3, 1].set_title("GM")

    plt.tight_layout()
    plt.show()
    
""" END Analysis of one single Subject """
# %%
    
    
""" Analysis-Loop for ALL Subjects """
# Pre-allocate dict for all CONTINUOUS data of all subjects
variables = c.variables
all_data_time = {}
all_data_angle = {}

for task in c.conditions["task"]:
    for muscle in c.conditions["muscle"]:
        for vib in c.conditions["vibration"]:
            
            condition_name = f"{task}_{muscle}_{vib}"
            
            all_data_time[condition_name] = {
                var: pd.DataFrame() for var in variables
            }
            
            all_data_angle[condition_name] = {
                var: pd.DataFrame() for var in variables
            }
            
""" Pre-allocate dict for all DISCRETE data of all subjects """
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
            
all_data_discrete = {}

for task in c.conditions["task"]:
    for muscle in c.conditions["muscle"]:
        for vib in c.conditions["vibration"]:

            condition_name = f"{task}_{muscle}_{vib}"

            all_data_discrete[condition_name] = pd.DataFrame(
                index=discrete_vars,
                dtype=float
            )
            

for subjID in subjIDs:
    
    matches = [item for item in c.rawDataDir.iterdir()
               if item.is_dir() and subjID in item.name]

    dominance = subjects["Dominance"][subjects.index[subjects['ID'] == subjID][0]]
    
    subjDir = matches[0] / "successful"

    ### Create folders if not already existant ###
    subj_process_path = c.processedDataDir / subjID
    subj_mvc_path = subj_process_path / "mvc_results"

    os.makedirs(subj_process_path, exist_ok=True)
    os.makedirs(subj_mvc_path, exist_ok=True)

    subjTrials = list(subjDir.glob("*.txt"))

    """ --- Rest Processing --- """
    rest_path = u.filter_files(subjTrials, "rest")
    rest = dp.rest_processing(rest_path, dominance)
    
    """ --- MVC load in --- """
    mvc_file = pd.read_pickle(subj_mvc_path / "mvc_data.pkl")
    mvc = mvc_file['mvc']
    
    """ Pre-allocate dicts for subject data """
    
    subject_data_time = {}
    subject_data_angle = {}
    subject_discrete_data = {}
    
    for task in c.conditions["task"]:
        for muscle in c.conditions["muscle"]:
            for vib in c.conditions["vibration"]:
                
                name = f"{task}_{muscle}_{vib}"
                subject_data_time[name] = {}
                subject_data_angle[name] = {}
                subject_discrete_data[name] = {}
    
    
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
            
        ### For angle normalization from 0 to 72.5 degrees (70 degrees as goal with 5% margin of error) 
        if "flex" in str(path):
            angle_grid = np.linspace(0, 72.5, 726) * -1
        elif "ext" in str(path):
            angle_grid = np.linspace(0, 72.5, 726)
        
        data, data_angle, data_discrete, test_test, bf_bp = dp.data_processing(path, condition, mvc, rest, dominance, angle_grid)
    
        subject_data_time[title] = data
        subject_data_angle[title] = data_angle
        subject_discrete_data[title] = data_discrete
        
    for condition_name, variables_dict in subject_data_time.items():
    
        for var, trials in variables_dict.items():
    
            mean_series = trials.mean(axis=1)
    
            all_data_time[condition_name][var][subjID] = mean_series
                        
    for condition_name, variables_dict in subject_discrete_data.items():
    
        for var, trials in variables_dict.items():
    
            mean_value = trials.mean(axis=1)
    
            all_data_discrete[condition_name].loc[var, subjID] = mean_value.iloc[0]
            
    for condition_name, variables_dict in subject_data_angle.items():
    
        for var, trials in variables_dict.items():
    
            mean_series = np.nanmean(trials, axis=1)
    
            all_data_angle[condition_name][var][subjID] = mean_series
""" END Analysis-Loop ALL Subejcts """

""" Extra Analysis """
# Calculate continuous ratios and store them under "ratio"
# all_data_ratio = {}
# all_data_diff_ham = {}
# all_data_diff_qh = {}

# for cond, dfs in all_data_time.items():

#     ham_mean = (dfs["BF"] + dfs["ST"]) / 2

#     all_data_ratio[cond] = {
#         "ratio": dfs["VL"] / ham_mean  if "ext" in cond else ham_mean / dfs["VL"]
#     }
#     all_data_diff_ham[cond] = {
#         "diff_ham": (dfs["BF"] - dfs["ST"]) / (dfs["BF"] + dfs["ST"])
#     }
#     all_data_diff_qh[cond] = {
#         "diff_qh": (dfs["VL"] - ham_mean) / (dfs["VL"] + ham_mean)
#     }
        
# t, ti, d1, d2 = run_spm(
#     data_dict=all_data_ratio,
#     condition1="30_flex_vib",
#     condition2="30_flex_non",
#     variable="ratio",
#     normalization="time"
# )
""" ENd extra Analysis """
       
# %%
""" Plot averaged data """
diff = "60"
task = "flex"
variables       = ["Pos", "BF", "Vel", "ST", "Acc", "VL", "Torq", "GM"]

# define fixed y-limits per variable
y_limits = {
    "BF": (0, 1),
    "ST": (0, 1),
    "VL": (0, 1),
    "GM": (0, 1),
    "Pos": (-80, 10),
    "Vel": (-5, 45),
    "Torq": (-0.05, 1),
    "Acc": (-100, 100)
}

# keys for conditions
key_non = f"{diff}_{task}_non"
key_vib = f"{diff}_{task}_vib"

fig, axes = plt.subplots(4, 2, constrained_layout=True)
axes = axes.flatten()

for i, var in enumerate(variables):
    ax = axes[i]
    
    data_non = all_data_time[key_non][var]
    data_vib = all_data_time[key_vib][var]
    
    mean_non = data_non.mean(axis=1)
    std_non = data_non.std(axis=1)
    
    mean_vib = data_vib.mean(axis=1)
    std_vib = data_vib.std(axis=1)
    
    # only label first subplot for shared legend
    label_non = "non" if i == 0 else None
    label_vib = "vib" if i == 0 else None
    
    ax.plot(mean_non, label=label_non, color="blue")
    ax.plot(mean_vib, label=label_vib, color="red")
    
    # shaded std
    ax.fill_between(mean_non.index, mean_non - std_non, mean_non + std_non, color="blue", alpha=0.2)
    ax.fill_between(mean_vib.index, mean_vib - std_vib, mean_vib + std_vib, color="red", alpha=0.2)
    
    ax.set_title(var)
    
    # set fixed y-limits
    ax.set_ylim(y_limits[var])
    
    # remove box around plots
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # x-axis only on bottom row
    if i < 6:  # not bottom row
        ax.set_xlabel("")
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
    else:
        ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
        ax.set_xticklabels(['0', '1', '2', '3', '4', '5'])
        ax.set_xlabel("Time [s]")

# shared legend
fig.legend(loc="upper right", fontsize=12)

# overall figure title
fig.suptitle(f"Task: {task}, Intensity: {diff}%"  + " (Time-Norm.)", fontsize=16)

fig.savefig(f"{task}_{diff}%" + "_TimeNorm" + ".svg", bbox_inches="tight")

plt.show()
# %%


# %%
""" Plot averaged data """
diff = "60"
task = "ext"
variables       = ["Pos", "BF", "Vel", "ST", "Acc", "VL", "Torq", "GM"]

t = np.linspace(0, 72.5, 726)

# define fixed y-limits per variable
y_limits = {
    "BF": (0, 1),
    "ST": (0, 1),
    "VL": (0, 1),
    "GM": (0, 1),
    "Pos": (-10, 80),
    "Vel": (-45, 5),
    "Torq": (-0.05, 1),
    "Acc": (-150, 100)
}

# keys for conditions
key_non = f"{diff}_{task}_non"
key_vib = f"{diff}_{task}_vib"

fig, axes = plt.subplots(4, 2, constrained_layout=True)
axes = axes.flatten()

for i, var in enumerate(variables):
    ax = axes[i]
    
    data_non = all_data_angle[key_non][var]
    data_vib = all_data_angle[key_vib][var]
    
    mean_non = data_non.mean(axis=1)
    std_non = data_non.std(axis=1)
    
    mean_vib = data_vib.mean(axis=1)
    std_vib = data_vib.std(axis=1)
    
    # only label first subplot for shared legend
    label_non = "non" if i == 0 else None
    label_vib = "vib" if i == 0 else None
    
    ax.plot(mean_non, label=label_non, color="blue")
    ax.plot(mean_vib, label=label_vib, color="red")
    
    # shaded std
    ax.fill_between(mean_non.index, mean_non - std_non, mean_non + std_non, color="blue", alpha=0.2)
    ax.fill_between(mean_vib.index, mean_vib - std_vib, mean_vib + std_vib, color="red", alpha=0.2)
    
    ax.set_title(var)
    #ax.set_ylabel("Value")
    
    # set fixed y-limits
    ax.set_ylim(y_limits[var])
    
    # remove box around plots
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # x-axis only on bottom row
    if i < 6:  # not bottom row
        ax.set_xlabel("")
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
    else:
        ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700])
        ax.set_xticklabels(['0', '10', '20', '30', '40', '50', '60', '70'])
        ax.set_xlabel("Angle [°]")

# shared legend
fig.legend(loc="upper right", fontsize=12)

# overall figure title
fig.suptitle(f"Task: {task}, Intensity: {diff}%" + " (Angle-Norm.)", fontsize=16)

#fig.savefig(f"{task}_{diff}%" + "_AngleNorm" + ".svg", bbox_inches="tight")

plt.show()
# %%
    




