# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt



def parameters(file_path, b=1, truncate=None):
    
    
    #Load Impulse Response file
    IR_raw, fs = sf.read(file_path)
    
    #Define band index array
    if b == 3:
        band_idx = np.arange(-16, 14)
    else:
        band_idx = np.arange(-5, 5)
    
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
        #Append parameters to lists
        C50.append(C50_from_IR(IR_filtered))
        C80.append(C80_from_IR(IR_filtered))
        IACCe.append(IACCe_from_IR(IR_filtered)
        #etc...
    
    return T10, T20, T30, C50, C80, IACCe, Tt, EDTt
        
        
    


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



def _Lundeby(IR):
    pass

def _Schroeder(IR):
    pass

def _movingMedian(IR):
    pass

