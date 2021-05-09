# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import butter, sosfilt, medfilt, correlate, fftconvolve
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
        self.EDTTt = []
        self.IACCe = []
        self.b = 1
        self.crossing_point = []
        self.fs = None
        self.t = None
        self.ETC_averaged = []




def parameters(IR_raw, fs, b=1, truncate=None, smoothing='schroeder', median_window=20, ignore_errors=False, verbose=False):
    
    param = AcParam()
    

    
    #Start at peak of impulse
    IR_raw = IR_raw[np.argmax(IR_raw):]
    
    #Trim
    IR_raw = trim_impulse(IR_raw, fs)
    
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
        
        #Trim last 5 percent of signal to minimize filter edge effect
        IR_filtered = IR_filtered[:round(len(IR_filtered)*0.95)]
        
        #Square (obtain Energy Time Curve ETC)
        ETC_band = IR_filtered ** 2
        #Normalize
        ETC_band = ETC_band / np.max(ETC_band)
        
        #dB and average (only for graph)
        ETC_dB_band = 10*np.log10(ETC_band/np.max(ETC_band))
        ETC_avg_dB_band = 10*np.log10(moving_average(ETC_band/np.max(ETC_band), 240))
        

        
        
        #Truncate
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
            ETC_averaged = False
        else:
            print('invalid truncate')
        
        #Smoothing
        if smoothing == 'schroeder':
            decay_band = schroeder(ETC_truncated_band, ETC_band.size-crossing_point_band)
        elif smoothing == 'median':
            decay_band = median_filter(ETC_truncated_band, f_low, fs, median_window, ETC_band.size-crossing_point_band)
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
        param.Tt.append(Tt_from_IR(ETC_truncated_band, fs))
        param.EDTTt.append(EDTTt_from_IR(decay_band, param.Tt[-1], fs))
       
       
        
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
    # t_2=int(0.001 * fs + 1) # 1ms samples
    band=np.arange(len(paramL.IR_filtered))
    
     
    for idx in band:
        pl = paramL.IR_filtered[(idx)][:(t_1)]
        pr = paramR.IR_filtered[(idx)][:(t_1)]
    
        pl_2 = pl ** 2
        pr_2 = pr ** 2
    
        IACF = correlate(pl, pr, method='fft') / np.sqrt(np.sum(pl_2) * np.sum(pr_2))
        IACC= np.amax(np.abs(IACF))
    
        paramL.IACCe.append(IACC)

    return paramL.IACCe
    
def Tt_from_IR(ETC, fs):
    idx_5ms = int(0.005 * fs)
    ETC = ETC[idx_5ms:]
    energy_total = np.sum(ETC)
    energy_cum = np.cumsum(ETC)

    Tt_idx = np.argmin(np.abs(energy_cum - 0.99*energy_total)) + idx_5ms
    Tt = Tt_idx / fs
    
    return Tt
    
    

def EDTTt_from_IR(signal, Tt, fs):
    s_init=int(0.005*fs)
    s_end = int(Tt * fs)
    factor=6

    
    #cut signal
    signal=signal[s_init:s_end]
    
    x = np.arange(s_init, s_end)
    y=signal
   
    # Linear regression
    slope, intercept =np.polyfit(x,y,1)
    line = slope * np.arange(signal.size) + intercept
    
    init = line[0]
    t = np.argmin(np.abs(line - (init - 10))) / fs
    
    EDTTt= factor * t
    
    return EDTTt


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
    # Invert
    IR = IR[-1:0:-1]
    
    nyq = 0.5 * fs
    if f_high >= nyq:
        f_high = nyq-1
    low = f_low / nyq
    high = f_high / nyq
    sos = butter(4, [low, high], btype="band", output="sos")
    IR_filtered = sosfilt(sos, IR)
    #Invert
    IR_filtered = IR_filtered[-1:0:-1]
    
    return IR_filtered

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



def lundeby(ETC, maf_window, band, fs, verbose=False):

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
    # Trim last 5 percent of signal for high frequencies
    if band > 8000:
        idx_last_5percent = ETC_trim.size-int(ETC_trim.size/20)
        ETC_trim = ETC_trim[:idx_last_5percent]
    
    
    #ETC_trim = ETC
    
    idx_last_10percent = -int(ETC_trim.size/10) #start index of last 10% of signal
    idx_last_5percent = ETC_trim.size-int(ETC_trim.size/20)
    
    #1) Moving average filter, window from 10 to 50ms
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

    
    
    #Start at peak of impulse
    ETC_averaged = ETC_averaged[np.argmax(ETC_averaged):]
    offset = len(ETC) - len(ETC_averaged)
        
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
    #Linear regression
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
    
    #Exception for too low dynamic range
    if dyn_range <= dB_to_noise + late_dyn_range:
        if verbose == True:
            print(band, "Hz band: Dynamic Range too low!")
        return crossing_point_pre + offset
    
    interval_num = np.intc(interval_density * dyn_range / 10)
    new_window = np.intc(idx_stop / interval_num)
    
    
    # 6) Moving average filter with new window
    ETC_averaged = moving_average(ETC, new_window)
    #Start at peak of impulse
    ETC_averaged = ETC_averaged[np.argmax(ETC_averaged):]
    offset = len(ETC) - len(ETC_averaged)
    #dB
    ETC_avg_dB = 10 * np.log10(ETC_averaged)

    
    
    # Iterate through steps 7), 8) and 9) until convergence
    crossing_point_old = crossing_point + 1000
    counter = 0
    
    
    while np.abs(crossing_point - crossing_point_old) > 0.005*fs:   #While difference in crossing points is larger than 1 ms
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

        # Exception for other rare errors
        if idx_stop <= idx_start:
            if verbose == True:
                print(band, "Hz: Regression failed (idx)")
            return crossing_point_pre+offset
        
        # Linear regression
        x = np.arange(idx_start, idx_stop)
        lin_reg = linregress(x, ETC_avg_dB[idx_start:idx_stop])
        line = lin_reg.slope * np.arange(ETC_avg_dB.size) + lin_reg.intercept
        
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
                                           

    #Truncate
    #ETC_truncated = ETC[:crossing_point]
    
    #return ETC_averaged, noise_estimate, lin_reg
    #return line, noise_estimate
    return crossing_point+offset
    

def schroeder(ETC, pad):
    # Schroeder integration
    sch = np.cumsum(ETC[::-1])[::-1]
    # Pad with zeros for same array length
    sch = np.concatenate((sch, np.zeros(pad)))  
    # dB scale, normalize
    with np.errstate(divide='ignore'): #ignore divide by zero warning
        sch = 10.0 * np.log10(sch / np.max(sch))


    return sch

def median_filter(ETC, f, fs, window, pad):
    
    #Convert window in milliseconds to an odd number of samples
    window = int(window/1000 * fs)
    if window % 2 == 0:
        window +=1
    
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

def trim_impulse(IR, fs, mode='IR'):
    # Trims the Impulse Response signal after 2 seconds of steady noise level
    
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
    IR = fftconvolve(sweep, inverse_filter, mode='same')
    IR = np.trim_zeros(IR)
    return IR
    
    