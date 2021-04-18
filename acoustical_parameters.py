# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, medfilt



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

    
    for idx in band_idx:
        #Band limits
        f_low, f_high = limits_iec_61260(idx, b)
        #Apply bandpass filter
        IR_filtered = bandpass(IR_raw, f_low, f_high, fs)
        #Square (obtain Energy Time Curve ETC)
        ETC_band = IR_filtered ** 2
        
        
        
        #Normalize
        #ETC_band = ETC_band / np.max(ETC_band)
        
        
        #Truncate
        if truncate == 'lundeby':
            ETC_band = lundeby(ETC_band)
        elif truncate != None:
            print('invalid truncate')
        
        #Smoothing
        if smoothing == 'schroeder':
            decay_band = schroeder(ETC_band)
        elif smoothing == 'median':
            decay_band = median_filter(ETC_band, f_low, fs)
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



def lundeby(ETC):
    return ETC
    

def schroeder(ETC):
    # Schroeder integration
    sch = np.cumsum(ETC[::-1])[::-1]
    # dB scale, normalize
    sch = 10.0 * np.log10(sch / np.max(sch))
    return sch

def median_filter(ETC, f, fs):
    #No sé qué hacer con la ventana. Tiene que ser impar
    median_window = 1501 #para 32 Hz
    # median_window = round(1 / f * fs)
    # if median_window % 2 == 0:
    #     median_window += 1
    # if median_window < 3:
    #     median_window = 3
    
    #Median filter
    med = medfilt(ETC, median_window)
    #dB scale, normalize
    med = 10*np.log10(med / np.max(med))
    
    return med

