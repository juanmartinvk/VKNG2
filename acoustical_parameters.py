# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, correlate, fftconvolve
from scipy.stats import linregress
from scipy.ndimage import median_filter as mmf

class AcParam:
# =============================================================================
#   An object of the AcParam class contains attributes related to the acoustical
#   parameters calculated from a Room Impulse Response file. For lists of parameters,
#   the first value correspond to the full band calculation, the others to the 
#   band-filtered parameters from valid frequency bands.
# =============================================================================
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
        self.EDTTt = []
        self.IACCe = []
        self.b = 1
        self.crossing_point = []
        self.fs = None
        self.nominalBands = []
        self.t = None
        self.ETC_averaged = []



def analyzeFile(impulsePath, filterPath, b, f_lim=(20, 20000), truncate=None, smoothing="schroeder", median_window=20):
    '''
    Reads a mono or stereo Impulse Response file and returns mono and stereo 
    acoustical parameters. The audio file can be input as an IR signal, or as a
    recorded sine sweep with its corresponding inverse filter.
    
    Parameters
    ----------
    impulsePath : str
        Impulse Response or recorded sine sweep audio file path.
    filterPath : str or None
        Inverse Filter audio file path for recorded sine sweep signals. If using
        Impulse Response signal, use None.
    b : int
        Octave fraction for filtering. Default is 1 for octave bands, use 3 for
        third-octave bands.
    f_lim : tuple, optional
        Frequency limits (bandwidth) of the analyzed signal in Hz. Default is 
        (20, 20000).
    truncate : str, optional
        IR truncation method. Default is None for no truncation. Use 'lundeby' for
        truncation by Lundeby's method.
    smoothing : str, optional
        ETC smoothing method. Default is 'schroeder' for Schroeder's Reverse
        Integration. Use 'median' for moving median filter.
    median_window : int, optional
        Window length for moving median filter in ms. Default is 20.

    Returns
    -------
    acParamL : AcParam object
        Left channel (or mono) acoustical parameters.
    acParamR : AcParam object or None
        Right channel acoustical parameters, or None for mono files.

    '''
    # Read file
    IR_raw, fs = sf.read(impulsePath)
    IR_raw_L = IR_raw
    IR_raw_R = None
    
    # Check if stereo or mono. If stereo, split channels.
    if IR_raw.ndim == 2:
        IR_raw_L = IR_raw[0:, 0]
        IR_raw_R = IR_raw[0:, 1]
    
    # If sweep, convolve with inverse filter
    if filterPath is not None:
        inverse_filter, fs_filter = sf.read(filterPath)
        if inverse_filter.ndim == 2:
            inverse_filter = inverse_filter[0:, 0]
        if fs != fs_filter:
            print("Sampling rates of sweep and inverse filter do not match")
        IR_raw_L = convolve_sweep(IR_raw_L, inverse_filter)
        if IR_raw_R is not None:
            IR_raw_R = convolve_sweep(IR_raw_R, inverse_filter)
    
    
    # Calculate parameters
    acParamL = parameters(IR_raw_L, fs, b, f_lim=f_lim, truncate=truncate, smoothing=smoothing, median_window=median_window)
    if IR_raw_R is not None:
        acParamR = parameters(IR_raw_R, fs, b, f_lim=f_lim, truncate=truncate, smoothing=smoothing, median_window=median_window)
        acParamR.IACCe=np.round(IACCe_from_IR(acParamL, acParamR), decimals=3)
        acParamL.IACCe=acParamR.IACCe
    else:
        acParamR = None

        
    return acParamL, acParamR



def parameters(IR_raw, fs, b=1, f_lim=(20, 20000), truncate=None, smoothing='schroeder', median_window=20, ignore_errors=False, verbose=False):
    '''
    Receives a mono Impulse Response signal and returns its monaural acoustical 
    parameters in the form of an object of the AcParam class.
    
    Parameters
    ----------
    IR_raw : 1-d array_like
        Impulse Response signal.
    fs : int
        IR sample rate.
    b : int, optional
        Octave fraction for filtering. Default is 1 for octave bands, use 3 for
        third-octave bands.
    f_lim : tuple, optional
        Frequency limits (bandwidth) of the analyzed signal in Hz. Default is 
        (20, 20000).
    truncate : str or None, optional
        Truncation method. Default is None for no truncation. Use 'lundeby' for
        truncation by Lundeby's method.
    smoothing : str, optional
        ETC smoothing method. Default is 'schroeder' for Schroeder's Reverse
        Integration. Use 'median' for moving median filter.
    median_window : int, optional
        Window length for moving median filter in ms. Default is 20.
    ignore_errors : boolean, optional
        Default is False. Use True to skip outlier detection.
    verbose : boolean, optional
        Default is False. Use True to print logs and error messages

    Returns
    -------
    param : AcParam object
        Calculated acoustical parameters (see AcParam documentation).

    '''
    
    # Initialize object of the AcParam class
    param = AcParam()
    
    #Start at peak of impulse
    IR_raw = IR_raw[np.argmax(IR_raw):]
    
    #Trim
    IR_raw = trim_impulse(IR_raw, fs)
    
    mmf_factor = median_window
    
    #Define band index array, nominal bands and band limits
    
    if b == 3:
        band_idx = np.arange(-16, 14)
        nominal_bands = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1300,
                         1600, 2000, 2500, 3200, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
        param.nominalBandsStr = ['25', '31.5', '40', '50', '63', '80', '100', '125', '160',
                         '200', '250', '315', '400', '500', '630', '800', '1k',
                         '1.3k', '1.6k', '2k', '2.5k', '3.2k', '4k', '5k', 
                         '6.3k', '8k', '10k', '12.5k', '16k', '20k']
        band_lim = [limits_iec_61260(idx, b) for idx in band_idx]
        mmf_windows = [round(mmf_factor/(1.2589**(x-1)))+3 for x in range(len(band_idx))]
    elif b == 1:
        band_idx = np.arange(-5, 5)
        nominal_bands = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        param.nominalBandsStr = ['31.5', '63', '125', '250', '500',
                         '1k', '2k', '4k', '8k', '16k']
        band_lim = [limits_iec_61260(idx, b) for idx in band_idx]
        mmf_windows = [round(mmf_factor/(2**x)+3) for x in range(len(band_idx))]
    else:
        print('invalid b')

    #Define window times for moving average filters (between 10 and 50 ms) for each band
    maf_windows = np.linspace(0.01, 0.05, num=len(band_idx)) #Generate times between 10 and 50ms
    maf_windows = np.intc(maf_windows * fs)[::-1] #Convert to integer number of samples and invert
    
    # Find band indexes excluded from bandwidth
    low_idx = 0
    high_idx = len(nominal_bands)
    for idx, limits in enumerate(band_lim):
        if limits[0] < f_lim[0]:
            low_idx = idx+1
        if limits[1] > f_lim[1]:
            high_idx = idx
            break
    
    # Trim unused bands
    band_idx = band_idx[low_idx:high_idx]
    nominal_bands = nominal_bands[low_idx:high_idx]
    band_lim = band_lim[low_idx:high_idx]
    param.nominalBandsStr = param.nominalBandsStr[low_idx:high_idx]
    maf_windows = maf_windows[low_idx:high_idx]
    mmf_windows = mmf_windows[low_idx:high_idx]
    



    
    # Full range parameters
    
    #Band limits
    f_low = f_lim[0]
    f_high = f_lim[1]
    #Apply bandpass filter
    IR_filtered = bandpass(IR_raw, f_low, f_high, fs)
    
    #Trim last 5 percent of signal to minimize filter edge effect
    IR_filtered = IR_filtered[:round(len(IR_filtered)*0.95)]
    
    #Square (obtain Energy Time Curve ETC)
    ETC_band = IR_filtered ** 2
    #Normalize
    ETC_band = ETC_band / np.max(ETC_band)
    
    #dB and average (only for plotting purposes)
    ETC_dB_band = 10*np.log10(ETC_band/np.max(ETC_band))
    ETC_avg_dB_band = 10*np.log10(moving_average(ETC_band/np.max(ETC_band), 240))
    
    #Truncate IR at crossing point
    if truncate == 'lundeby':
        try:
            crossing_point_band = lundeby(ETC_band, maf_windows[0], f_low, fs, verbose=verbose)
        except:
            if verbose == True:
                print("Unknown error in truncation")
            crossing_point_band = ETC_band.size
        ETC_truncated_band = ETC_band[:crossing_point_band]
    elif truncate is None:
        ETC_truncated_band = ETC_band
        crossing_point_band = ETC_band.size
    else:
        print('invalid truncate')
    
    #Smoothing
    if smoothing == 'schroeder':
        decay_band = schroeder(ETC_truncated_band, ETC_band.size-crossing_point_band)
    elif smoothing == 'median':
        #decay_band = median_filter(ETC_truncated_band, fs, median_window, ETC_band.size-crossing_point_band)
        decay_band = median_filter(ETC_truncated_band, fs, mmf_windows[0], ETC_band.size-crossing_point_band)
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
    param.Tt.append(Tt_from_IR(ETC_truncated_band, fs, f_low))
    param.EDTTt.append(EDTTt_from_IR(decay_band, param.Tt[-1], fs))
    # param.EDTTt.append(EDTTt_from_IR(decay_band, 0.2, fs))

    
    # Band-filtered parameters
    
    counter = 0
    for idx in band_idx:
        #Band limits
        f_low = band_lim[counter][0]
        f_high = band_lim[counter][1]
        #Apply bandpass filter
        IR_filtered = bandpass(IR_raw, f_low, f_high, fs)
        
        #Trim last 5 percent of signal to minimize filter edge effect
        IR_filtered = IR_filtered[:round(len(IR_filtered)*0.95)]
        
        #Square (obtain Energy Time Curve ETC)
        ETC_band = IR_filtered ** 2
        #Normalize
        ETC_band = ETC_band / np.max(ETC_band)
        
        #dB and average (only for plotting purposes)
        ETC_dB_band = 10*np.log10(ETC_band/np.max(ETC_band))
        ETC_avg_dB_band = 10*np.log10(moving_average(ETC_band/np.max(ETC_band), 240))
        
        #Truncate IR at crossing point
        if truncate == 'lundeby':
            try:
                crossing_point_band = lundeby(ETC_band, maf_windows[counter], nominal_bands[counter], fs, verbose=verbose)
            except:
                if verbose == True:
                    print("Unknown error in truncation")
                crossing_point_band = ETC_band.size
            ETC_truncated_band = ETC_band[:crossing_point_band]
        elif truncate is None:
            ETC_truncated_band = ETC_band
            crossing_point_band = ETC_band.size
        else:
            print('invalid truncate')
        
        #Smoothing
        if smoothing == 'schroeder':
            decay_band = schroeder(ETC_truncated_band, ETC_band.size-crossing_point_band)
        elif smoothing == 'median':
            decay_band = median_filter(ETC_truncated_band, fs, mmf_windows[counter], ETC_band.size-crossing_point_band)
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
        param.Tt.append(Tt_from_IR(ETC_truncated_band, fs, f_low))
        param.EDTTt.append(EDTTt_from_IR(decay_band, param.Tt[-1], fs))
        # param.EDTTt.append(EDTTt_from_IR(decay_band, 0.2, fs))
        
        counter += 1
    
    
    #Round parameters to 2 decimals
    param.EDT = np.round(param.EDT, 2)
    param.T20 = np.round(param.T20, 2)
    param.T30 = np.round(param.T30, 2)
    param.C50 = np.round (param.C50, 2)
    param.C80 = np.round(param.C80, 2)
    param.Tt = np.round(param.Tt, 2)
    param.EDTTt = np.round(param.EDTTt, 2)
    
    #Add fs and time axis to param
    param.fs = fs
    param.t = np.linspace(0, param.ETC[0].size / param.fs , num=param.ETC[0].size)
    
    # Identify outliers and replace with zeros:
    if ignore_errors == False:
        EDT_error = [0] * len(nominal_bands)
        T20_error = [0] * len(nominal_bands)
        T30_error = [0] * len(nominal_bands)
        EDTTt_error = [0] * len(nominal_bands)
        for i in range(len(nominal_bands)):
            if param.EDT[i] > 5*np.median(param.EDT):
                EDT_error[i] = 1
            if param.T20[i] > 5*np.median(param.T20):
                T20_error[i] = 1
            if param.T30[i] > 5*np.median(param.T30):
                T30_error[i] = 1       
            if param.EDTTt[i] > 5*np.median(param.EDTTt):
                EDTTt_error[i] = 1
        for i in range(len(nominal_bands)):
            if EDT_error[i] == 1:
                param.EDT[i] = 0
            if T20_error[i] == 1:
                param.T20[i] = 0
            if T30_error[i] == 1:
                param.T30[i] = 0
            if EDTTt_error[i] == 1:
                param.EDTTt[i] = 0
    
    # Return param object
    return param
        
        


def EDT_from_IR(decay, fs):
    '''
    Calculates Early Decay Time parameter from decay curve.

    Parameters
    ----------
    decay : 1-d array_like
        Decay curve (truncated and smoothed ETC).
    fs : int
        Sample rate.

    Returns
    -------
    EDT : float
        Calculated Early Decay Time value.

    '''
    
    # Define start and end levels, time factor
    init=0
    end=-10
    factor=6
    
    # Start at peak (useful for smoothing methods other than Schroeder)
    decay = decay[np.argmax(decay):]
    
    # Find start and end samples
    s_init = np.argmin(np.abs(decay - init))
    s_end = np.argmin(np.abs(decay-end))
    
    # Slice
    decay=decay[s_init:s_end]
    
    # Define axes for linear regression
    t = np.arange(s_init, s_end) / fs
    y=decay
   
    # Linear regression
    slope, intercept =np.polyfit(t,y,1)

    # Find line values at start and end samples
    init_value=(init-intercept)/slope
    end_value=(end-intercept)/slope
    
    # Calculate EDT
    EDT= factor * (end_value-init_value)
    
    return EDT

def T20_from_IR(decay, fs):
    '''
    Calculates T20 parameter from decay curve.

    Parameters
    ----------
    decay : 1-d array_like
        Decay curve (truncated and smoothed ETC).
    fs : int
        Sample rate.

    Returns
    -------
    T20 : float
        Calculated T20 value.

    '''
    # Define start and end levels, time factor
    init=-5
    end=-25
    factor=3
    
    # Start at peak (useful for smoothing methods other than Schroeder)
    decay = decay[np.argmax(decay):]
    
    # Find start and end samples
    s_init = np.argmin(np.abs(decay - init))
    s_end = np.argmin(np.abs(decay-end))
    
    # Slice
    decay=decay[s_init:s_end]
    
    # Define axes for linear regression
    t = np.arange(s_init, s_end) / fs
    y=decay
   
    # Linear regression
    slope, intercept =np.polyfit(t,y,1)

    # Find line values at start and end samples
    init_value=(init-intercept)/slope
    end_value=(end-intercept)/slope
    
    # Calculate T20
    T20= factor * (end_value-init_value)
    
    return T20



def T30_from_IR(decay, fs):
    '''
    Calculates T30 parameter from decay curve.

    Parameters
    ----------
    decay : 1-d array_like
        Decay curve (truncated and smoothed ETC).
    fs : int
        Sample rate.

    Returns
    -------
    T30 : float
        Calculated T30 value.

    '''
    init=-5
    end=-35
    factor=2
    
    # Start at peak (useful for smoothing methods other than Schroeder)
    decay = decay[np.argmax(decay):]
    
    # Find start and end samples
    s_init = np.argmin(np.abs(decay - init))
    s_end = np.argmin(np.abs(decay-end))
    
    # Slice
    decay=decay[s_init:s_end]
    
    # Define axes for linear regression
    t = np.arange(s_init, s_end) / fs
    y=decay
   
    # Linear regression
    slope, intercept =np.polyfit(t,y,1)

    # Find line values at start and end samples
    init_value=(init-intercept)/slope
    end_value=(end-intercept)/slope
    
    # Calculate EDT
    T30= factor * (end_value-init_value)
    
    return T30


def C50_from_IR(fs, ETC):
    '''
    Calculates C50 value from Energy Time Curve.

    Parameters
    ----------
    fs : int
        Sample rate.
    ETC : 1-d array_like
        Energy Time Curve signal.

    Returns
    -------
    C50 : float
        Calculated C50 value.

    '''
    # Find sample value for 50 ms 
    t = int(0.05 * fs + 1)
    
    # Calculate C50
    C50= 10.0 * np.log10((np.sum(ETC[:t]) / np.sum(ETC[t:])))
    
    return C50 



def C80_from_IR(fs, ETC):
    '''
    Calculates C80 value from Energy Time Curve.

    Parameters
    ----------
    fs : int
        Sample rate.
    ETC : 1-d array_like
        Energy Time Curve signal.

    Returns
    -------
    C80 : float
        Calculated C80 value.

    '''
    
    # Find sample value for 80 ms
    t = int(0.08 * fs + 1)
    
    #Calculate C80
    C80= 10.0 * np.log10((np.sum(ETC[:t]) / np.sum(ETC[t:])))
    
    return C80 

def IACCe_from_IR(paramL, paramR):
    '''
    Calculate Inter-Aural Cross-Correlation (Early) values from left and right
    channel acoustical parameters    
    
    Parameters
    ----------
    paramL : AcParam object
        Parameters for left channel.
    paramR : AcParam object
        Parameters for right channel.

    Returns
    -------
    IACCe : list
        IACC (Early) values for each octave or third-octave band.

    '''
    
    # Find sample value for 80 ms
    fs=paramL.fs
    t_1= int(0.08 * fs + 1) # 80ms samples
    
    # Define band index list
    bands = np.arange(len(paramL.IR_filtered))
     
    for idx in bands:
        
        # Slice left and right IR signals
        pl = paramL.IR_filtered[idx][:t_1]
        pr = paramR.IR_filtered[idx][:t_1]
        
        # Square (ETC)
        pl_2 = pl ** 2
        pr_2 = pr ** 2
        
        # Define Inter-Aural Cross-Correlation Function
        IACF = correlate(pl, pr, method='fft') / np.sqrt(np.sum(pl_2) * np.sum(pr_2))
        
        # Calculate IACC
        IACC= np.amax(np.abs(IACF))
        
        # Append to list
        paramL.IACCe.append(IACC)

    return paramL.IACCe
    

def Tt_from_IR(ETC, fs, f_low):
    '''
    Calculates Transition Time from Energy Time Curve.

    Parameters
    ----------
    ETC : 1-d array_like
        Energy Time Curve signal.
    fs : int
        Sample rate.
    f_low : int
        Lower frequency limit.

    Returns
    -------
    Tt : float
        Calculated Transition Time value in seconds.

    '''
    
    if f_low > 100:
        window = round(fs/f_low)
        # print(window)
        if window % 2 == 0:
            window = window + 1
        window = int(window)
        ETC_median = median_filter(ETC, fs, window, 0)
        # ETC_median = median_filter(ETC, fs, 51, 0)
        
        DCER = 10*np.log10(ETC) + ETC_median
        DCER = 10**(DCER/10)
        
        actual_EDF = np.cumsum(DCER)
        actual_EDF = actual_EDF/np.max(actual_EDF)
        
        
        Tt = np.argmin(np.abs(actual_EDF-0.99*np.max(actual_EDF)))/fs
    else:
        Tt = 0
    
    # Tt = 1
    
    return Tt
    
    

def EDTTt_from_IR(decay, Tt, fs):
    '''
    Calculate Early Decay Time (Transition Time) parameter. This value estimates
    the decay rate between the start of the impulse and the room's Transition Time.

    Parameters
    ----------
    decay : 1-d array_like
        Decay curve (truncated and smoothed ETC).
    Tt : float
        Transition Time in seconds.
    fs : int
        Sample rate.

    Returns
    -------
    EDTTt : float
        Calculated Early Decay Time (Transition Time) value.

    '''
    if Tt > 0.01:
    # if False:
    # Define start and end levels, time factor
        s_init=int(0.005*fs)
        s_end = int(Tt * fs)
        factor=6
    
        
        # Slice
        decay=decay[s_init:s_end]
        
        # Define axes for linear regression
        x = np.arange(s_init, s_end)
        y=decay
       
        # Linear regression
        slope, intercept =np.polyfit(x,y,1)
        line = slope * np.arange(decay.size) + intercept
        
        # Find the time it takes for the signal to decay 10 dB
        init = line[0]
        t = np.argmin(np.abs(line - (init - 10))) / fs
        
        #Calculate EDTTt
        EDTTt= factor * t
    else:
        EDTTt = 0
    
    return EDTTt


def bandpass(IR, f_low, f_high, fs):
    '''
    Applies an 8th order bandpass filter to a signal within the desired frequencies.

    Parameters
    ----------
    IR : 1-d array_like
        Raw Impulse Response signal.
    f_low : float
        Lower frequency limit.
    f_high : float
        Upper frequency limit.
    fs : int
        Sample Rate.

    Returns
    -------
    IR_filtered : 1-d array_like
         Filtered Impulse Response
    '''
    
    # Invert IR
    IR = IR[-1:0:-1]
    
    # Find Nyquist Frequency
    nyq = 0.5 * fs
    
    # Limit upper frequency limit to Nyquist frequency
    if f_high >= nyq:
        f_high = nyq-1
    
    # Define filter parameters as second-order sections
    low = f_low / nyq
    high = f_high / nyq
    sos = butter(4, [low, high], btype="band", output="sos")
    
    # Apply filter
    IR_filtered = sosfilt(sos, IR)
    
    #Invert
    IR_filtered = IR_filtered[-1:0:-1]
    
    return IR_filtered

def limits_iec_61260(index, b, fr=1000):
    """
    Calculates low and high band limits of a nominal band of a specific index
    with respect to a reference frequency, as per IEC 61260-1:2014
    
    Parameters
    ----------
    index : int
        Band index with respect to reference frequency
    b : int
        Band filter fraction. E.G. for third-octave, b=3
    fr : float
        Reference frequency. The default is 1000 Hz.
        
    Returns
    -------
    f_lim : tuple
        Tuple with the form (f_low, f_high), frequency band limits for filter

    """
    # Define G
    G=10**(3/10)
    
    #Obtain exact center frequency from index
    if b % 2 == 0:
        f_center = G**((2*index+1)/(2*b)) * fr
    else:
        f_center = G**(index/b) * fr
    
    #Calculate band limits
    f_low = G**(-1/(2*b)) * f_center
    f_high = G**(1/(2*b)) * f_center
    
    f_lim = (f_low, f_high)
    return f_lim



def lundeby(ETC, maf_window, band, fs, verbose=False):
    '''
    Obtains the crossing point index of a Room Impulse Response using Lundeby's
    method.

    Parameters
    ----------
    ETC : 1-d array_like
        Energy Time Curve signal.
    maf_window : list
        List of windows for moving average filters according to band index.
    band : float
        Frequency band.
    fs : int
        Sample rate.
    verbose : boolean, optional
        Default is False. Use True to print logs and error messages

    Returns
    -------
    crossing_point : int
        Index of RIR crossing point.

    '''
    
    
    late_dyn_range = 15 # Dynamic range to be used for late decay slope estimation
    dB_to_noise = 7 # dB above noise for linear regression
    
    #Define number of time intervals per each 10 dB of dynamic range for MAF
    if band <= 32:
        interval_density = 2
    elif band <= 63:
        interval_density = 2
    elif band <= 160:
        interval_density = 3
    elif band <= 500:
        interval_density = 3
    elif band <= 2000:
        interval_density = 5
    elif band <= 8000:
        interval_density = 7
    else:
        interval_density = 10
    
    # Trim excess noise tail (more than 2 seconds)
    ETC_trim = trim_impulse(ETC, fs, mode='ETC')
    
    # Trim last 5 percent of signal for high frequencies (to avoid edge effects)
    if band > 8000:
        idx_last_5percent = ETC_trim.size-int(ETC_trim.size/20)
        ETC_trim = ETC_trim[:idx_last_5percent]
    
    # Define indexes for last 10% and 5% of the signal
    idx_last_10percent = -int(ETC_trim.size/10)
    idx_last_5percent = ETC_trim.size-int(ETC_trim.size/20)
    
    #1) Moving average filter. Windows manually set in bands below 200 Hz
    # (this does not follow Lundeby's recommendations, but proved more effective)
    if band > 200:
        ETC_averaged = moving_average(ETC_trim, maf_window)
    elif band > 160:
        ETC_averaged = moving_average(ETC_trim, 2000)
    elif band > 63:
        ETC_averaged = moving_average(ETC_trim, 3000)
    elif band > 32:
        ETC_averaged = moving_average(ETC_trim, 6000)
    else:
        ETC_averaged = moving_average(ETC_trim, 10000)

    
    # Start at peak of impulse
    ETC_averaged = ETC_averaged[np.argmax(ETC_averaged):]
    # Store offset generated by previous line
    offset = len(ETC) - len(ETC_averaged)
    # dB
    ETC_avg_dB = 10 * np.log10(ETC_averaged)
    

    # 2) Estimate noise level with last 10% of the signal
    noise_estimate = 10 * np.log10( np.mean(ETC_averaged[idx_last_10percent:]) )
    # Exception for REALLY LOW dynamic range
    if np.max(ETC_avg_dB) <= dB_to_noise + noise_estimate:
        if verbose == True:
            print(band, "Hz band: This doesn't look like a Room Impulse Response!")
        return ETC.size
    
    
    # 3) Estimate preliminar slope
    idx_stop = np.where(ETC_avg_dB >= noise_estimate + dB_to_noise)[0][-1]
    x = np.arange(idx_stop)
    # Linear regression
    lin_reg = linregress(x, ETC_avg_dB[:idx_stop])
    line = lin_reg.slope * np.arange(ETC_avg_dB.size) + lin_reg.intercept

    # 4) Find preliminar crossing point
    crossing_point_pre = np.argmin(np.abs(line - noise_estimate))
    crossing_point = crossing_point_pre
    if crossing_point >= ETC.size or crossing_point <= np.argmax(ETC):
        if verbose == True:
            print(band, "Hz band: Regression failed (pre)")
        return ETC.size
    
    # 5) Calculate new interval length for moving average filter
    dyn_range = lin_reg.intercept - noise_estimate
    
    # Exception for too low dynamic range
    if dyn_range <= dB_to_noise + late_dyn_range:
        if verbose == True:
            print(band, "Hz band: Dynamic Range too low!")
        return crossing_point_pre + offset
    
    # Calculate number of intervals for new windows
    interval_num = np.intc(interval_density * dyn_range / 10)
    # Calculate new window for MAF
    new_window = np.intc(idx_stop / interval_num)
    
    # 6) Moving average filter with new window
    ETC_averaged = moving_average(ETC, new_window)
    # Start at peak of impulse
    ETC_averaged = ETC_averaged[np.argmax(ETC_averaged):]
    offset = len(ETC) - len(ETC_averaged)
    # dB
    ETC_avg_dB = 10 * np.log10(ETC_averaged)

    
    # Iterate through steps 7), 8) and 9) until convergence
    crossing_point_old = crossing_point + 1000
    counter = 0
    
    while np.abs(crossing_point - crossing_point_old) > 0.005*fs:   #While difference in crossing points is larger than 1 ms    
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

        # Exception for other rare errors
        if idx_stop <= idx_start:
            if verbose == True:
                print(band, "Hz: Regression failed (idx)")
            return crossing_point_pre+offset
        
        # Linear regression
        x = np.arange(idx_start, idx_stop)
        lin_reg = linregress(x, ETC_avg_dB[idx_start:idx_stop])
        line = lin_reg.slope * np.arange(ETC_avg_dB.size) + lin_reg.intercept
        
        # Exceptions for invalid line slopes
        if lin_reg.slope > 0:
            if verbose == True:
                print(band, "Hz: Positive slope!")
            ETC_averaged = True
            return crossing_point_pre + offset
        
        if lin_reg.slope == 0:
            if verbose == True:
                print(band, "Hz: Flat line!")
            return crossing_point_pre + offset
        
        # 9) Find new crosspoint
        crossing_point = np.argmin(np.abs(line - noise_estimate))
        if crossing_point+offset <= np.argmax(ETC):
            if verbose == True:
                print(band, "Hz: Regression failed (x-point less than max)")
            ETC_averaged = True
            return crossing_point_pre+offset
        
        # Exception for crossing point too close to end of signal
        if (counter > 5 and crossing_point > ETC_averaged.size+idx_last_10percent) or crossing_point > idx_last_5percent :
            if crossing_point_pre > idx_last_10percent:    
                if verbose == True:
                    print(band, "Hz band: crosspoint too close to end")
                return ETC.size
            
        #Exception for too many loops
        if counter > 30:
            if verbose == True:
                print(band, 'Hz: Could not achieve convergence. Abort!')
            if abs(crossing_point-crossing_point_old) > 0.01:
                crossing_point = min(crossing_point_pre, crossing_point)
            ETC_averaged = True
            break
        
        counter += 1
                                           
    # Add offset to crossing point
    crossing_point = crossing_point + offset
    
    return crossing_point
    

def schroeder(ETC, pad):
    '''
    Applies Schroeder's Reverse Integration to the truncated Energy Time Curve.

    Parameters
    ----------
    ETC : 1-d array_like
        Energy Time Curve signal.
    pad : int
        Amount of padding necessary to return same array length as original 
        (non-truncated) ETC signal.

    Returns
    -------
    sch : 1-d array_like
        Schroeder integrated decay curve.

    '''
    # Schroeder integration
    sch = np.cumsum(ETC[::-1])[::-1]
    # Pad with zeros for same array length as original ETC (for plotting purposes)
    sch = np.concatenate((sch, np.zeros(pad)))  
    # dB scale, normalize
    with np.errstate(divide='ignore'): #ignore divide by zero warning
        sch = 10.0 * np.log10(sch / np.max(sch))

    return sch

def median_filter(ETC, fs, window, pad):
    '''
    Applies a moving median filter with the desired window length.
    
    Parameters
    ----------
    ETC : 1-d array_like
        Energy Time Curve signal.
    fs : int
        Sample rate.
    window : float
        Window length in milliseconds.
    pad : int
        Amount of padding necessary to return same array length as original 
        (non-truncated) ETC signal.

    Returns
    -------
    med : 1-d array_like
        Median-filtered decay curve.

    '''
    
    #Convert window in milliseconds to an odd number of samples
    window = int(window/1000 * fs)
    if window % 2 == 0:
        window +=1
    
    #Median filter
    med = mmf(ETC, size=window, mode="nearest")
    # Pad with zeros for same array length
    med = np.concatenate((med, np.zeros(pad)))  
    #dB scale, normalize
    with np.errstate(divide='ignore'): #ignore divide by zero warning
        med = 10*np.log10(med / np.max(med))

    return med

def moving_average(ETC, window):
    '''
    Applies a moving average filter with the desired window length.

    Parameters
    ----------
    ETC : 1-d array_like
        Energy Time Curve signal.
    window : int
        Window length in samples.

    Returns
    -------
    ETC_averaged : 1-d array_like
        Averaged ETC.

    '''
    ETC_padded = np.pad(ETC, (window//2, window-1-window//2), mode='edge') #Pad with edge values
    ETC_averaged = np.convolve(ETC_padded, np.ones((window,))/window, mode='valid') #Moving average filter

    return ETC_averaged

def trim_impulse(IR, fs, mode='IR'):
    '''
    Trims the Impulse Response signal after 2 seconds of steady noise level.

    Parameters
    ----------
    IR : 1-d array_like
        Room Impulse Response signal (or Energy Time Curve).
    fs : int
        Sample rate.
    mode : str, optional
        Use 'IR' if using an Impulse Response, use 'ETC' for an Energy Time Curve.
        The default is 'IR'.

    Returns
    -------
    IR: 1-d array_like
        Trimmed Impulse Response.

    '''
    # Trim zeros
    IR = np.trim_zeros(IR)
    #Average response and dB
    with np.errstate(divide='ignore'): #Ignore divide by zero warning
        if mode == 'IR':
            ETC_dB = 10 * np.log10(moving_average(IR**2, 500))
        elif mode == 'ETC':
            ETC_dB = 10 * np.log10(moving_average(IR, 500))
    
    # Define chunk size
    chunk_t = 0.5
    chunk_samples = int(chunk_t * fs)
    chunk_num = int(IR.size // chunk_samples) #Number of whole chunks in IR
    
    # Exception for very short IR
    if chunk_num <= 3:
        return IR
    
    
    mean_old = 0
    mean_new = 0
    same_counter = 0
    
    # Sweep the signal in 1 second chunks with 50% overlap. 
    # After 4 iterations of average level within +-3dB, truncate the signal at that point
    for i in range(chunk_num-1):
        
        mean_old = mean_new
        
        if same_counter == 4:
            return IR[:(i-1)*chunk_samples]
        
        start = i * chunk_samples
        stop = (i + 2) * chunk_samples
        mean_new = np.mean(ETC_dB[start:stop])
        
        #Exception for bumps
        if mean_new > mean_old + 3:
            return IR[:(i-1)*chunk_samples]
        
        #If within 3 dB of the previous chunk, add 1 to counter. If not, reset.
        if mean_new > mean_old - 3:
            same_counter += 1
        else:
            same_counter = 0
    
    return IR
        

def convolve_sweep(sweep, inverse_filter):
    '''
    Convolves a recorded sine sweep with its respective inverse filter.

    Parameters
    ----------
    sweep : 1-d array_like
        Recorded sine sweep
    inverse_filter :  1-d array_like
        Inverse filter.

    Returns
    -------
    IR :  1-d array_like
        Calculated Impulse Response.

    '''
    IR = fftconvolve(sweep, inverse_filter, mode='same')
    IR = np.trim_zeros(IR)
    return IR
    
    