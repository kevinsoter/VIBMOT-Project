# -*- coding: utf-8 -*-
"""
SPM paired comparison for time- or angle-normalized data

Author: Kevin Soter
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import spm1d
from scipy.interpolate import interp1d

# Downsample waveform

def downsample_curve(y, n_points):
    """
    Downsample a 1D waveform by interpolation.

    Parameters
    ----------
    y : array-like
        Original waveform.

    n_points : int
        Desired number of points.

    Returns
    -------
    ndarray
    """

    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, n_points)

    f = interp1d(x_old, y, kind="linear")

    return f(x_new)


# Main SPM function

def run_spm(
    data_dict,
    condition1,
    condition2,
    variable,
    normalization="angle",
    alpha=0.05,
    savefig=False,
    savepath=None,
):

    """
    Parameters
    ----------
    data_dict :
        all_data_time or all_data_angle

    condition1 :
        e.g. "30_flex_vib"

    condition2 :
        e.g. "30_flex_non"

    variable :
        "Torq","Vel","BF","ST",...

    normalization :
        "time" or "angle"

    """

    # number of nodes

    if normalization.lower() == "time":

        n_nodes = 501
        x = np.linspace(0, 100, n_nodes)
        xlabel = "Movement duration (%)"

    elif normalization.lower() == "angle":

        n_nodes = 281

        # if you prefer degrees:
        x = np.linspace(0, 70, n_nodes)
        xlabel = "Joint angle (°)"

        # if you prefer %ROM instead:
        # x = np.linspace(0,100,n_nodes)
        # xlabel="Range of motion (%)"

    else:

        raise ValueError("normalization must be 'time' or 'angle'")

    # Extract DataFrames

    df1 = data_dict[condition1][variable].copy()
    df2 = data_dict[condition2][variable].copy()

    # Common subjects

    common_subjects = sorted(
        list(
            set(df1.columns).intersection(df2.columns)
        )
    )

    if len(common_subjects) == 0:

        raise ValueError("No common subjects found.")

    df1 = df1[common_subjects]
    df2 = df2[common_subjects]

    # Downsample every subject

    data1 = []
    data2 = []

    for subj in common_subjects:

        y1 = downsample_curve(df1[subj].values, n_nodes)
        y2 = downsample_curve(df2[subj].values, n_nodes)

        data1.append(y1)
        data2.append(y2)

    data1 = np.vstack(data1)
    data2 = np.vstack(data2)

    # Remove NaNs

    valid = (
        ~np.isnan(data1).any(axis=1)
        &
        ~np.isnan(data2).any(axis=1)
    )

    data1 = data1[valid]
    data2 = data2[valid]

    subjects = np.array(common_subjects)[valid]

    print("---------------------------------------")
    print("Variable:", variable)
    print("Normalization:", normalization)
    print("Condition 1:", condition1)
    print("Condition 2:", condition2)
    print("Subjects:", len(subjects))
    print("---------------------------------------")

    # Mean +/- SEM

    mean1 = np.mean(data1, axis=0)
    mean2 = np.mean(data2, axis=0)

    sem1 = np.std(data1, axis=0, ddof=1) / np.sqrt(data1.shape[0])
    sem2 = np.std(data2, axis=0, ddof=1) / np.sqrt(data2.shape[0])
    
    ci1 = 1.96 * sem1
    ci2 = 1.96 * sem2

    # SPM

    t = spm1d.stats.ttest_paired(
        data1,
        data2
    )

    ti = t.inference(
        alpha=alpha,
        two_tailed=True
    )

    # Plot
    
    fig, axs = plt.subplots(
        2,
        1,
        figsize=(10,6),
        sharex=True,
        gridspec_kw={"height_ratios":[2,1]}
    )
    
    # x-axis
    
    if normalization.lower() == "time":
    
        x = np.arange(n_nodes)
    
        tick_pos = np.linspace(0, n_nodes-1, 6)
        tick_lab = ["0","20","40","60","80","100"]
    
        xlabel = "Movement duration (%)"
    
    else:
    
        x = np.arange(n_nodes)
    
        tick_pos = np.linspace(0, n_nodes-1, 8)
        tick_lab = [f"{v:.0f}" for v in np.linspace(0,70,8)]
    
        xlabel = "Joint angle (°)"
    
    
    # Mean curves
    color1 = "red"
    color2 = "blue"
    
    
    ax = axs[0]
    
    ax.plot(x, mean2,
            color=color2,
            lw=2,
            label=condition2)
    
    ax.fill_between(
        x,
        mean2-sem2,
        mean2+sem2,
        color=color2,
        alpha=0.25
    )
    
    # upper/lower SEM boundaries
    # ax.plot(
    #     x,
    #     mean2 + sem2,
    #     color=color2,
    #     lw=0.5,
    #     )

    # ax.plot(
    #     x,
    #     mean2 - sem2,
    #     color=color2,
    #     lw=0.5,
    # )
    
    ax.plot(x, mean1,
            color=color1,
            lw=2,
            ls="--",
            label=condition1)
    
    ax.fill_between(
        x,
        mean1-sem1,
        mean1+sem1,
        color=color1,
        alpha=0.25,
        edgecolor="none"
    )
    
    # upper/lower SEM boundaries
    # ax.plot(
    #     x,
    #     mean1 + sem1,
    #     color=color1,
    #     lw=0.5,
    #     ls="--",
    #     )

    # ax.plot(
    #     x,
    #     mean1 - sem1,
    #     color=color1,
    #     lw=0.5,
    #     ls="--",
    # )
    
    ax.legend()
    ax.set_ylabel(variable)
    ax.set_ylim(-0.0, 0.9)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    
    # SPM
    
    ax2 = axs[1]
    
    ti.plot(ax=ax2)
    ti.plot_threshold_label(ax=ax2)
    ti.plot_p_values(ax=ax2)
    
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_lab)
    
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("t")
    ax2.set_ylim(-6.5, 6.5)
    
    plt.tight_layout()
    plt.show()

    return t, ti, data1, data2

t, ti, d1, d2 = run_spm(
    data_dict=all_data_diff_qh,
    condition1="30_ext_vib",
    condition2="30_ext_non",
    variable="diff_qh",
    normalization="time"
)

# %%
# Getting peak t values per cluster
for i, c in enumerate(ti.clusters):

    # Cluster boundaries (continuum indices)
    start = int(np.floor(c.endpoints[0]))
    end   = int(np.ceil(c.endpoints[1]))
    
    threshold   = c.threshold

    # t-values within this cluster
    tvals = ti.z[start:end+1]

    if threshold > 0:
        # Peak t-value
        peak_t = tvals.max()
    
        # Location of the peak
        peak_idx = (start + np.argmax(tvals)) / 500 * 100
    elif threshold < 0:
        # Peak t-value
        peak_t = tvals.min()
    
        # Location of the peak
        peak_idx = (start + np.argmin(tvals)) / 500 * 100

    print(f"Cluster {i+1}")
    print(f"  Cluster t theshold = {threshold:.3f}")
    print(f"  Cluster p = {c.P:.4f}")
    print(f"  Endpoints = {start / 500 * 100: .1f} to {end / 500 * 100: .1f}")
    print(f"  Peak t = {peak_t:.3f}")
    print(f"  Peak location = {peak_idx: .1f}%")
    print(f"  Endpoints Exact = {start: .1f} to {end: .1f}")
    
# %%
    
# Save result 
# import pickle    
# with open("all_data_diff_qh.pkl", "wb") as f:
#     pickle.dump(all_data_diff_qh, f)
    
# # Open result 
# import pickle    
# with open("all_data_time.pkl", "rb") as f:
#     data_time = pickle.load(f)

    
    