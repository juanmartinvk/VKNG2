# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import acoustical_parameters as ap
import soundfile as sf
import numpy as np
from scipy.ndimage import median_filter as mmf
from scipy.optimize import curve_fit
import dictances as dc

def bhattacharyya_distance(distribution1, distribution2):
    """ Estimate Bhattacharyya Distance (between General Distributions)
    
    Args:
        distribution1: a sample distribution 1
        distribution2: a sample distribution 2
    
    Returns:
        Bhattacharyya distance
    """
    sq = 0
    for i in range(len(distribution1)):
        sq  += np.sqrt(distribution1[i]*distribution2[i])
    print(sq)
    
    return -np.log(sq)


def return_cap_charge(origin):
    def cap_charge(t, Tt):
        a = 1 - origin
        b = 4.60517*(1/Tt)
        # b = 4*(1/Tt)
        cap = 1-a*np.exp(-b * t)
        return cap
    
    return cap_charge

def median_filter(ETC, fs, window, pad):
   
    #Convert window in milliseconds to an odd number of samples
    window = int(window/1000 * fs)
    if window % 2 == 0:
        window +=1
    
    #Median filter
    med = mmf(ETC, size=window, mode="nearest")

    return med

def IR_to_filteredETC(IR_raw, fs, f_low, f_high):
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
            
         #Truncate IR at crossing point
        crossing_point_band = ap.lundeby(ETC_band, 2000, 1000, fs)
        ETC_band = ETC_band[:crossing_point_band]  
        
        return ETC_band

def tex_curves(ETC, fs, f_low):
    
        # Median Filter
        window = int(round(fs/f_low))
        if window % 2 == 0:
            window = window + 1
        
        ETC_median = median_filter(ETC, fs, window, 0)
        DCER = 10*np.log10(ETC) + ETC_median
        DCER = 10**(DCER/10)
        
        #Calculate actual Echo Density Function
        actual_EDF = np.cumsum(DCER)
        actual_EDF = actual_EDF/np.max(actual_EDF)
        
        # Calculate Transition Time 
        Tt_idx = np.argmin(np.abs(actual_EDF-0.99*np.max(actual_EDF)))
        Tt = Tt_idx/fs
        
        # Compute ideal EDF
        t = np.linspace(0, len(ETC)/fs, len(ETC))
        # t = np.logspace(0, len(ETC)/fs, len(ETC))
            
        cap_charge = return_cap_charge(actual_EDF[0])
        ideal_EDF = np.array([cap_charge(x, Tt) for x in t])
        
        # Compute expected EDF
        popt, pcov = curve_fit(cap_charge, t, actual_EDF)
        expected_EDF = np.array([cap_charge(x, popt[0]) for x in t])
        
        return actual_EDF, ideal_EDF, expected_EDF, DCER, Tt_idx
    
def band_texture(actual_EDF, ideal_EDF, expected_EDF, Tt_idx):
        # ideal_dic = dict(enumerate(ideal_EDF.flatten(), 1))
        # expected_dic = dict(enumerate(expected_EDF.flatten(), 1))
        
        # DBM_band = dc.bhattacharyya(expected_dic,ideal_dic)
        
        actual_EDF = actual_EDF[:Tt_idx]
        ideal_EDF = ideal_EDF[:Tt_idx]
        expected_EDF = expected_EDF[:Tt_idx]
        
        DBM_band = bhattacharyya_distance(expected_EDF,ideal_EDF)
        ETx_band = np.corrcoef(expected_EDF, actual_EDF)[0][1]
        Tx_band = np.corrcoef(ideal_EDF, actual_EDF)[0][1]
        
        return Tx_band, ETx_band, DBM_band

def texture(filename, b):
    
    ETx = []
    Tx = []
    DBM = []
    
    IR_raw, fs = sf.read(filename)
    if IR_raw.ndim == 2:
        IR_raw = IR_raw[0:, 0]
    
    
    if b == 3:
        band_idx = np.arange(-16, 14)
    elif b == 1:
        band_idx = np.arange(-5, 5)
        
    
    for band in range(len(band_idx)):
    # band = 5
        f_low, f_high = ap.limits_iec_61260(band_idx[band], b)
        
        if f_low < 100:
            ETx.append(0)
            Tx.append(0)
            DBM.append(0)
            
        else:
            
            ETC_band = IR_to_filteredETC(IR_raw, fs, f_low, f_high)
            
            actual_EDF, ideal_EDF, expected_EDF, _, Tt_idx = tex_curves(ETC_band, fs, f_low)
            
            Tx_band, ETx_band, DBM_band = band_texture(actual_EDF, ideal_EDF, expected_EDF, Tt_idx)
            
            ETx.append(ETx_band)
            Tx.append(Tx_band)
            DBM.append(DBM_band)
        
    return ETx, Tx, DBM



