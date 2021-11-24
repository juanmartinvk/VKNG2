
# -*- coding: utf-8 -*-
import acoustical_parameters as ap
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import time
import anisotropia as an
from scipy.ndimage import median_filter as mmf

def median_filter(ETC, fs, window, pad):

    #Convert window in milliseconds to an odd number of samples
    window = int(window/1000 * fs)
    if window % 2 == 0:
        window +=1
    
    #dB scale, normalize
    with np.errstate(divide='ignore'): #ignore divide by zero warning
        ETC = 10*np.log10(ETC)
        
    #Median filter
    med = mmf(ETC, size=window, mode="nearest")

    return med

b = 1

if b == 3:
    band_idx = np.arange(-16, 14)
elif b == 1:
    band_idx = np.arange(-5, 5)
    
band = 5

f_low, f_high = ap.limits_iec_61260(band_idx[band], b)

IR_raw, fs = sf.read("sp1_mp1_ir.wav")


ETC_band = an.IR_to_filteredETC(IR_raw, fs, f_low, f_high)


ETC_median = median_filter(ETC_band, fs, 201, 0)

actual_EDF, ideal_EDF, expected_EDF, DCER = an.tex_curves(ETC_band, fs, f_low)

Tx, ETx, DBM = an.band_texture(actual_EDF, ideal_EDF, expected_EDF)

# Calculate Transition Time 
Tt = np.argmin(np.abs(actual_EDF-0.99*np.max(actual_EDF)))/fs

print("Tt: "+str(Tt))
print("Tx: "+str(Tx))
print("ETx: "+str(ETx))
print("DBM: "+str(DBM))


# after = time.time()
# Median filter
# decay_band = ap.median_filter(ETC_band, fs, mmf_windows[band], 0)

# plt.plot(ETC_median)
# plt.plot(10*np.log10(ETC_band))


plt.xscale("log")

# plt.plot(DCER)
plt.plot(actual_EDF)
plt.plot(ideal_EDF)
plt.plot(expected_EDF)