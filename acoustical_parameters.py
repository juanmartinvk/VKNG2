# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, medfilt
from scipy.stats import linregress
from statistics import mean



def parameters(file_path, b=1, truncate=None, smoothing='schroeder'):
    
    
    #Load Impulse Response file
    IR_raw, fs = sf.read(file_path)

    #Start at peak of impulse
    IR_raw = IR_raw[np.argmax(IR_raw):]
    

    
    #Define band index array
    if b == 3:
        band_idx = np.arange(-16, 14)
    elif b == 1:
        band_idx = np.arange(-5, 5)
    else:
        print('invalid b')
    
    #Define window times for moving average filters (between 10 and 50 ms) for each band

    maf_windows = np.linspace(0.01, 0.05, num=len(band_idx)) #Generate times between 10 and 50ms
    maf_windows = np.intc(maf_windows * fs)[::-1] #Convert to integer number of samples and invert

    
    ETC = []
    decay = []
    T10 = []
    T20 = []
    T30 = []
    C50 = []
    C80 = []
    IACCe = []
    Tt = []
    EDTt = []

    maf_window_idx = 0
    for idx in band_idx:
        #Band limits
        f_low, f_high = limits_iec_61260(idx, b)
        #Apply bandpass filter
        IR_filtered = bandpass(IR_raw, f_low, f_high, fs)
        #Square (obtain Energy Time Curve ETC)
        ETC_band = IR_filtered ** 2
        
        
        #Normalize
        ETC_band = ETC_band / np.max(ETC_band)
        
        
        #Truncate
        if truncate == 'lundeby':
            ETC_truncated, crossing_point = lundeby(ETC_band, maf_windows[maf_window_idx])
        elif truncate is None:
            ETC_truncated = ETC_band
        else:
            print('invalid truncate')
        
        #Smoothing
        if smoothing == 'schroeder':
            decay_band = schroeder(ETC_truncated, len(ETC_band)-crossing_point)
        elif smoothing == 'median':
            decay_band = median_filter(ETC_truncated, f_low, fs)
        else:
            print('invalid smoothing')

        
        
        #Append parameters to lists
        #A los parámetros que no requieran la integración,  se les puede
        #pasar directamente ETC_band o IR_filtered. Hay que ver qué onda
        #para IACCe
        ETC.append(ETC_band)     
        decay.append(decay_band)
        #C50.append(C50_from_IR(decay_band))
        #C80.append(C80_from_IR(decay_band))
        #etc...
        
        maf_window_idx += 1
        
    return ETC, decay
    #return T10, T20, T30, C50, C80, IACCe, Tt, EDTt
        
        
    


def RT_from_IR(IR, truncate):
    """ 

    """
    pass



def C50_from_IR(IR):
    pass

def C80_from_IR(IR):
    pass

def IACCe_from_IR(IR):
    pass

def Tt_from_IR(IR):
    pass

def EDTt_from_IR(IR):
    pass

def bandpass(IR, f_low, f_high, fs):
    """
    Applies a bandpass filter to a signal within the desired frequencies.

    Parameters
    ----------
    IR : Impulse response array.
    f_low : Low cut frequency.
    f_high : High cut frequency
    fs : Sample rate

    Returns
    -------
    Filtered impulse response array

    """
    nyq = 0.5 * fs
    if f_high >= nyq:
        f_high = nyq-1
    low = f_low / nyq
    high = f_high / nyq
    sos = butter(4, [low, high], btype="band", output="sos")    
    return sosfilt(sos, IR)

def limits_iec_61260(index, b, fr=1000):
    """
    Calculates low and high band limits, as per IEC 61260-1:2014

    Parameters
    ----------
    index : Band index with respect to reference frequency
    b : Band filter fraction. E.G. for third-octave, b=3
    fr : Reference frequency. The default is 1000 Hz.

    Returns
    -------
    f_low : Low band limit
    f_high : High band limit
    """
    G=10**(3/10)
    #Obtain exact center frequency from index
    if b % 2 == 0:
        f_center = G**((2*index+1)/(2*b)) * fr
    else:
        f_center = G**(index/b) * fr
    
    #Calculate band limits
    f_low = G**(-1/(2*b)) * f_center
    f_high = G**(1/(2*b)) * f_center
    
    return f_low, f_high



def lundeby(ETC, maf_window):
    dB_to_noise = 7 # dB above noise for linear regression
    
    # 1) Moving average filter, window from 10 to 50ms

    ETC_averaged = moving_average(ETC, maf_window)
    ETC_avg_dB = 10 * np.log10(ETC_averaged)
    
    # 2) Estimate noise level with last 10% of the signal
    noise_estimate = 10 * np.log10( np.mean(ETC_averaged[-int(ETC_averaged.size/10):]) )
    
    # 3) Estimate preliminar slope and crossing point
    stop_idx = np.where(ETC_avg_dB >= noise_estimate + dB_to_noise)[0][-1]
    x = np.arange(stop_idx)
    lin_reg = linregress(x, ETC_avg_dB[:stop_idx])
    line = lin_reg.slope * np.arange(ETC_avg_dB.size) + lin_reg.intercept
    crossing_point = np.argmin(np.abs(line - noise_estimate))
    
    

    
    #Truncate
    ETC_truncated = ETC[:crossing_point]
    
    #return ETC_averaged, noise_estimate, lin_reg
    return ETC_truncated, crossing_point
    

def schroeder(ETC, pad):
    # Schroeder integration
    sch = np.cumsum(ETC[::-1])[::-1]
    # Pad with zeros for same array length
    sch = np.concatenate((sch, np.zeros(pad)))    
    # dB scale, normalize
    sch = 10.0 * np.log10(sch / np.max(sch))

    return sch

def median_filter(ETC, f, fs):
    #No sé qué hacer con la ventana. Tiene que ser impar
    window = 1501 #para 32 Hz
    # window = round(1 / f * fs)
    # if window % 2 == 0:
    #     window += 1
    # if window < 3:
    #     window = 3
    
    #Median filter
    med = medfilt(ETC, window)
    #dB scale, normalize
    med = 10*np.log10(med / np.max(med))
    
    return med

def moving_average(ETC, window):    
    ETC_padded = np.pad(ETC, (window//2, window-1-window//2), mode='edge') #Pad with edge values
    ETC_averaged = np.convolve(ETC_padded, np.ones((window,))/window, mode='valid') #Moving average filter

    return ETC_averaged

