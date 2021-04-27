# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import butter, sosfilt, medfilt, correlate
from scipy.stats import linregress

class AcParam:
    def __init__(self):
        self.IR_filtered = []
        self.ETC = []
        self.ETC_dB = []
        self.ETC_avg_dB = []
        self.ETC_truncated = []
        self.decay = []
        self.EDT = []
        self.T20 = []
        self.T30 = []
        self.C50 = []
        self.C80 = []
        self.Tt = []
        self.EDTt = []
        self.IACCe = []
        self.b = 1
        self.crossing_point = []
        self.fs = None
        self.t = None




def parameters(IR_raw, fs, b=1, truncate=None, smoothing='schroeder'):
    
    param = AcParam()
    

    
    #Start at peak of impulse
    IR_raw = IR_raw[np.argmax(IR_raw):]

    
    #Define band index array and nominal bands
    if b == 3:
        band_idx = np.arange(-16, 14)
        nominal_bands = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1300, 1600, 2000, 2500, 3200, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
    elif b == 1:
        band_idx = np.arange(-5, 5)
        nominal_bands = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    else:
        print('invalid b')
    
    #Define window times for moving average filters (between 10 and 50 ms) for each band

    maf_windows = np.linspace(0.01, 0.05, num=len(band_idx)) #Generate times between 10 and 50ms
    maf_windows = np.intc(maf_windows * fs)[::-1] #Convert to integer number of samples and invert


    counter = 0
    for idx in band_idx:
        #Band limits
        f_low, f_high = limits_iec_61260(idx, b)
        #Apply bandpass filter
        IR_filtered = bandpass(IR_raw, f_low, f_high, fs)
        #Square (obtain Energy Time Curve ETC)
        ETC_band = IR_filtered ** 2
        #Normalize
        ETC_band = ETC_band / np.max(ETC_band)
        
        #dB and average (only for graph)
        ETC_dB_band = 10*np.log10(ETC_band/np.max(ETC_band))
        ETC_avg_dB_band = 10*np.log10(moving_average(ETC_band/np.max(ETC_band), 240))
        # ETC_avg_dB_band = moving_average(ETC_band, 240)
        # ETC_avg_dB_band =10*np.log10( ETC_avg_dB_band / np.max(ETC_avg_dB_band))
        
        
        #Truncate
        if truncate == 'lundeby':
            ETC_truncated_band, crossing_point_band = lundeby(ETC_band, maf_windows[counter], nominal_bands[counter], fs)
        elif truncate is None:
            ETC_truncated_band = ETC_band
            crossing_point_band = ETC_band.size
        else:
            print('invalid truncate')
        
        #Smoothing
        if smoothing == 'schroeder':
            decay_band = schroeder(ETC_truncated_band, ETC_band.size-crossing_point_band)
        elif smoothing == 'median':
            decay_band = median_filter(ETC_truncated_band, f_low, fs, ETC_band.size-crossing_point_band)
        else:
            print('invalid smoothing')

        
        
        #Append parameters to lists
        param.IR_filtered.append(IR_filtered)
        param.crossing_point.append(crossing_point_band)
        param.ETC.append(ETC_band)
        param.ETC_dB.append(ETC_dB_band)
        param.ETC_avg_dB.append(ETC_avg_dB_band)     
        param.decay.append(decay_band)
        param.EDT.append(EDT_from_IR(decay_band, fs))
        param.T20.append(T20_from_IR(decay_band, fs))
        param.T30.append(T30_from_IR(decay_band, fs))
        param.C50.append(C50_from_IR(fs, ETC_band))
        param.C80.append(C80_from_IR(fs, ETC_band))
       
       
        
        counter += 1
    
    #Round parameters to 2 decimals
    param.EDT = np.round(param.EDT, decimals=2)
    param.T20 = np.round(param.T20, decimals=2)
    param.T30 = np.round(param.T30, decimals=2)
    param.C50 = np.round (param.C50, decimals=2)
    param.C80 = np.round(param.C80, decimals=2)
    
    #Add fs and time axis to param
    param.fs = fs
    param.t = np.linspace(0, param.ETC[0].size / param.fs , num=param.ETC[0].size)
    
    #return ETC_avg_dB, decay, EDT, T20, T30, C50, C80
    #return  IACCe, Tt, EDTt
    return param
        
        
    


def EDT_from_IR(signal, fs):
# signal is the smoothed and truncated IR

    init=0
    end=-10
    factor=6
    
    signal = signal[np.argmax(signal):]
    s_init = np.argmin(np.abs(signal - init))
    s_end = np.argmin(np.abs(signal-end))
    
    #cut signal
    signal=signal[s_init:s_end]
    
    t = np.arange(s_init, s_end) / fs
    y=signal
   
    # Linear regression
    slope, intercept =np.polyfit(t,y,1)
    # EDT_aprox=np.polyval([slope, intercept],t)  #recta

    init_value=(init-intercept)/slope
    end_value=(end-intercept)/slope
    
    EDT= factor * (end_value-init_value)
    
    return EDT

def T20_from_IR(signal, fs):
# signal is the smoothed and truncated IR

    init=-5
    end=-25
    factor=3
    
    signal = signal[np.argmax(signal):]
    s_init = np.argmin(np.abs(signal - init))
    s_end = np.argmin(np.abs(signal-end))
    
    #cut signal
    signal=signal[s_init:s_end]      
    
    t = np.arange(s_init, s_end) / fs
    y=signal
   
    # Linear regression
    slope, intercept =np.polyfit(t,y,1)
    # T20_aprox=np.polyval([slope, intercept],t)  #recta

    init_value=(init-intercept)/slope
    end_value=(end-intercept)/slope
    
    T20= factor * (end_value-init_value)
    
    return T20



def T30_from_IR(signal, fs):
# signal is the smoothed and truncated IR

    init=-5
    end=-35
    factor=2
    
    signal = signal[np.argmax(signal):]
    s_init = np.argmin(np.abs(signal - init))
    s_end = np.argmin(np.abs(signal-end))
    
    #cut signal
    signal=signal[s_init:s_end]      
    
    t = np.arange(s_init, s_end) / fs
    y=signal
   
    # Linear regression
    slope, intercept =np.polyfit(t,y,1)
    # T30_aprox=np.polyval([slope, intercept],t)  #recta

    init_value=(init-intercept)/slope
    end_value=(end-intercept)/slope
    
    T30= factor * (end_value-init_value)
    
    return T30


def C50_from_IR(fs, ETC):
    
    t = int(0.05 * fs + 1) # 50ms samples
    C50= 10.0 * np.log10((np.sum(ETC[:t]) / np.sum(ETC[t:])))
    
    return C50 



def C80_from_IR(fs, ETC):
    
    t = int(0.08 * fs + 1) # 80ms samples
    C80= 10.0 * np.log10((np.sum(ETC[:t]) / np.sum(ETC[t:])))
    
    return C80 

def IACCe_from_IR(paramL, paramR):
   
    fs=paramL.fs
    t_1= int(0.08 * fs + 1) # 80ms samples
    t_2=int(0.001 * fs + 1) # 1ms samples
    band=np.arange(len(paramL.IR_filtered))
    
     
    for idx in band:
        pl = paramL.IR_filtered[(idx)][:(t_1)]
        pr = paramR.IR_filtered[(idx)][:(t_1)]
    
        pl_2 = paramL.ETC[(idx)][:(t_1)]
        pr_2 = paramR.ETC[(idx)][:(t_1)]
    
        IACF = correlate(pl, pr, method='fft') / np.sqrt(np.sum(pl_2) * np.sum(pr_2))
        IACC= np.amax(np.abs(IACF[:(t_2)]))
    
        paramL.IACCe.append(IACC)
    
    return paramL.IACCe
    
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



def lundeby(ETC, maf_window, band, fs):
    dB_to_noise = 7 # dB above noise for linear regression
    interval_density = 5 #number of time intervals per each 10 dB of dynamic range
    idx_last_10percent = -int(ETC.size/10) #start index of last 10% of signal
    late_dyn_range = 20 # Dynamic range to be used for late decay slope estimation
    
    # 1) Moving average filter, window from 10 to 50ms
    ETC_averaged = moving_average(ETC, maf_window)
    ETC_avg_dB = 10 * np.log10(ETC_averaged)
    
    # 2) Estimate noise level with last 10% of the signal
    noise_estimate = 10 * np.log10( np.mean(ETC_averaged[idx_last_10percent:]) )
    # Exception for REALLY LOW dynamic range
    if np.max(ETC_avg_dB) <= dB_to_noise + noise_estimate:
        print(band, "Hz band: This doesn't look like a Room Impulse Response!")
        return ETC, ETC.size
    
    # 3) Estimate preliminar slope
    idx_stop = np.where(ETC_avg_dB >= noise_estimate + dB_to_noise)[0][-1]
    x = np.arange(idx_stop)
    #Linear regression
    lin_reg = linregress(x, ETC_avg_dB[:idx_stop])
    line = lin_reg.slope * np.arange(ETC_avg_dB.size) + lin_reg.intercept

    # 4) Find preliminar crossing point
    crossing_point = np.argmin(np.abs(line - noise_estimate))
    
    # 5) Calculate new interval length for moving average filter
    dyn_range = lin_reg.intercept-noise_estimate
    
    #Exception for too low dynamic range
    if dyn_range <= dB_to_noise + late_dyn_range:
        print(band, "Hz band: Dynamic Range too low!")
        return ETC, ETC.size
    
    interval_num = np.intc(interval_density * dyn_range / 10)
    new_window = np.intc(idx_stop / interval_num)
    
    
    # 6) Moving average filter with new window
    ETC_averaged = moving_average(ETC, new_window)
    ETC_avg_dB = 10 * np.log10(ETC_averaged)
    
    
    # Iterate through steps 7), 8) and 9) until convergence
    crossing_point_old = crossing_point + 1000
    counter = 0
    
    
    while np.abs(crossing_point - crossing_point_old) > 0.001*fs:   #While difference in crossing points is larger than 1 ms
        #plt.plot(line)        
        crossing_point_old = crossing_point
        
        # 7) Estimate background noise level    
        #Allow a safety margin from crosspoint corresponding to 5-10 dB decay, but use a minimum of 10% of the impulse response.
        idx_10dB_below_crosspoint = np.argmin(np.abs(line - noise_estimate - 10))
        if idx_10dB_below_crosspoint >= idx_last_10percent:
            noise_estimate = 10 * np.log10(np.mean(ETC_averaged[idx_last_10percent:]))
        else:
            noise_estimate = 10 * np.log10(np.mean(ETC_averaged[idx_10dB_below_crosspoint:]))
        
        # 8) Estimate late decay slope
        # A dynamic range of 10-20 dB should be evaluated, starting 5-10 dB above the noise level.
        
        idx_start = np.argmin(np.abs(line - (noise_estimate + dB_to_noise + late_dyn_range) ))
        idx_stop = np.argmin(np.abs(line - (noise_estimate + dB_to_noise) ))

        #Linear regression
        x = np.arange(idx_start, idx_stop)
        lin_reg = linregress(x, ETC_avg_dB[idx_start:idx_stop])
        line = lin_reg.slope * np.arange(ETC_avg_dB.size) + lin_reg.intercept

        # 9) Find new crosspoint
        crossing_point = np.argmin(np.abs(line - noise_estimate))
        #Exception for too many loops
        if counter > 30:
            print(band, 'Hz: Could not achieve convergence. Abort!')
            crossing_point = ETC.size
            break
        
        counter += 1
                                           

    #Truncate
    ETC_truncated = ETC[:crossing_point]
    
    #return ETC_averaged, noise_estimate, lin_reg
    #return line, noise_estimate
    return ETC_truncated, crossing_point
    

def schroeder(ETC, pad):
    # Schroeder integration
    sch = np.cumsum(ETC[::-1])[::-1]
    # Pad with zeros for same array length
    sch = np.concatenate((sch, np.zeros(pad)))  
    # dB scale, normalize
    with np.errstate(divide='ignore'): #ignore divide by zero warning
        sch = 10.0 * np.log10(sch / np.max(sch))


    return sch

def median_filter(ETC, f, fs, pad):

    window = 2145
    #window = int( (1/f * fs) // 2 * 2 + 1)
    #Median filter
    med = medfilt(ETC, window)
    # Pad with zeros for same array length
    med = np.concatenate((med, np.zeros(pad)))  
    #dB scale, normalize
    with np.errstate(divide='ignore'): #ignore divide by zero warning
        med = 10*np.log10(med / np.max(med))

    return med

def moving_average(ETC, window):    
    ETC_padded = np.pad(ETC, (window//2, window-1-window//2), mode='edge') #Pad with edge values
    ETC_averaged = np.convolve(ETC_padded, np.ones((window,))/window, mode='valid') #Moving average filter

    return ETC_averaged

def convolve_sweep(sweep, inverse_filter):
    return sweep
