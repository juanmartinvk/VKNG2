# -*- coding: utf-8 -*-
import acoustical_parameters as ap
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import time

b = 1

if b == 3:
    band_idx = np.arange(-16, 14)
elif b == 1:
    band_idx = np.arange(-5, 5)
    
band = 8

f_low, f_high = ap.limits_iec_61260(band_idx[band], b)

IR_raw, fs = sf.read("sp1_mp1_ir.wav")

#Start at peak of impulse
IR_raw = IR_raw[np.argmax(IR_raw):]

#Trim
IR_raw = ap.trim_impulse(IR_raw, fs)

IR_filtered = ap.bandpass(IR_raw, f_low, f_high, fs)

#Trim last 5 percent of signal to minimize filter edge effect
IR_filtered = IR_filtered[:round(len(IR_filtered)*0.95)]

#Square (obtain Energy Time Curve ETC)
ETC_band = IR_filtered ** 2
#Normalize
ETC_band = ETC_band / np.max(ETC_band)


#median_window = 200

if b == 1:
    mmf_windows = [round(800/(2**x)+3) for x in range(len(band_idx))]
elif b == 3:
    mmf_windows = [round(1007/(1.2589**x))+3 for x in range(len(band_idx))]
print(mmf_windows)

after = time.time()
# Median filter
decay_band = ap.median_filter(ETC_band, fs, mmf_windows[band], 0)

plt.plot(decay_band)