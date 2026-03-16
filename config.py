# -*- coding: utf-8 -*-

# placeholder for constants
from pathlib import Path


'''--- Project Directories ---'''
projectRoot = Path(__file__).resolve().parent

dataDir = projectRoot / "data"
rawDataDir = dataDir / "raw"
processedDataDir = dataDir / "processed"
resultsDir = projectRoot / "results"

'''--- Constants ---'''
fs = 5000 # sampling rate

#conditions = ["mvc", "1RM", "30_vib", "30_non", "60_vib", "60_non"]

muscles = ["BF", "ST", "VL", "GM"]

conditions = {
    "task": ["30", "60"],
    "muscle": ["flex", "ext"],
    "vibration": ["vib", "non"]
}

'''--- Processing constants ---'''
bp_low = 20 # lowpass from bandpass filter / actually highpass and vice versa
bp_high = 450 # highpass from bandpass filter

lpCutoff = 6 # envelope filter
window_ms = 250

filtOrder = 2

convFact = 0.0136677881 # Conversion factor voltage to torque (HUMAC); Averag eof 10 subjects







