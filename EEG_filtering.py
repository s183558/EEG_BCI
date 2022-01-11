# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:31:00 2022

@author: Frederik
"""



##############################################################################
#                               Imports                                      #
##############################################################################


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Bandpass filter
# from scipy.signal import butter, sosfilt, sosfreqz



# Plotting
from scipy import signal


##############################################################################
#                              Classes                                       #
##############################################################################








##############################################################################
#                             Functions                                      #
##############################################################################

def load_openBCI_data(file_name : str):
    
    tic = time.perf_counter()
    
    # Read the first lines of the file, which is just info for the data
    with open(file_name) as myfile:
        head = [next(myfile) for x in range(4)]
    
    # Retrieve the number of channels and sameple rate from the data file 
    nb_chans    = int(head[1][head[1].find('channels =')+11:head[1].find('\n')])
    sample_rate = int(head[2][head[2].find('Rate =')+7:head[2].find('Hz')-1])
    
    
    # Reading the actual data and saving it in a pd DataFrame
    raw_data    = pd.read_csv(file_name, skiprows = 4, delimiter = ', ',
                              engine = 'python')
    
    # Drop the columns we dont need
    for col in raw_data.columns:
        if   'Other' in col:
            raw_data.drop(col, axis = 1, inplace = True)
            
        elif 'Accel' in col:
            raw_data.drop(col, axis = 1, inplace = True)
            
        elif 'Analog' in col:
            raw_data.drop(col, axis = 1, inplace = True)
            
        elif 'Sample2' in col:
            raw_data.drop(col, axis = 1, inplace = True)
            
        elif 'Formatted2' in col:
            raw_data.drop(col, axis = 1, inplace = True)
    
    
    
    
    
    
    toc = time.perf_counter()
    print(f'\nTime it took to load the data was {toc - tic:0.2f} seconds\n')
    
    
    return nb_chans, sample_rate, raw_data


# def butter_bandpass(lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
#     return sos


# def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#     sos = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = sosfilt(sos, data)
#     return y


def notch_filter(data, sample_rate : int, notch_freqs : [],
                 quality_factor = 20.0, show_plot = False):
    
    # Loop for all the notch frequncies that needs to be filtered out
    for notch_freq in notch_freqs:
        # Designing a notch filter using signal.iirnotch
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, sample_rate)
        
        # Apply notch filter to the noisy signal using signal.filtfilt
        data = signal.filtfilt(b_notch, a_notch, data)
    
    
    if show_plot:
        plot_FFT(data, sample_rate = sample_rate, lowest_freq = lowcut,
                 title = f'Notch @ {notch_freqs}Hz - FFT')
        
    return data
    

def plot_eeg(EEG_data, vspace=0, channels = 6, color='k', samplerate = 250):
    '''
    Plot the EEG data, stacking the channels horizontally on top of each other.

    Parameters
    ----------
    EEG_data : array (channels x samples)
        The EEG data
    vspace : float (default 100)
        Amount of vertical space to put between the channels
    channels : int (default 6)
        Amount of  channels in the data set
    color : string (default 'k')
        Color to draw the EEG in
        
    Ripped from:
    https://notebook.community/joannekoong/neuroscience_tutorials/basic/1.%20Load%20EEG%20data%20and%20plot%20ERP
    '''
    
    global chan_names
    # vspace * 0, vspace * 1,  ..., vspace * channels;
    bases = (vspace * np.arange(channels)).reshape(-1,1)
    
    # Center the data at 0
    EEG = EEG_data - EEG_data.mean(axis = 1, keepdims = 1)
    
    # To add the bases (a vector of length 7) to the EEG (a 2-D Matrix), we don't use
    # loops, but rely on a NumPy feature called broadcasting:
    # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    EEG += bases
    
    # Calculate a timeline in seconds, based on the EEG's sample rate
    time = np.arange(EEG.shape[1]) / samplerate
    
    # Plot EEG versus time
    plt.plot(time, EEG.T, color=color)

    # Add gridlines to the plot
    plt.grid()
    
    # Label the axes
    plt.xlabel('Time (s)')
    plt.ylabel('Channels')
    
    # The y-ticks are set to the locations of the electrodes. The international 10-20 system defines
    # default names for them.
    plt.gca().yaxis.set_ticks(bases.flatten())
    plt.gca().yaxis.set_ticklabels(chan_names[:channels])
    
    # Put a nice title on top of the plot
    plt.title('EEG data')
    
    plt.show()


def plot_FFT(data, sample_rate : int, lowest_freq : float, title : str):
    """
    # Plot the FFT plot from OpenBCI

    # Parameters
    # ----------
    # data : 2D np.array
    #     An array of the data for the channels which are wanted to be plotted.
    # sample_rate : int
    #     The sample rate.
    # lowest_freq : float
    #     The lowest frequency in the data.
    # title : str
    #     A title for the plot.

    """
    
    global chan_names
    # The lowest time window to analyze the signal is to have atleast 2 cycles of 
    # the lowest frequency measured n the data, which would be 0.5 Hz, as that
    # is the lowest delta signal. 
    win = (2/lowest_freq) * sample_rate
    
    if max(data.shape) < win:
        raise ValueError(f'Input data must contain enough samples to satisfy '\
                         f'the Nyqueist theorem. win = {win} <= '\
                         f'samples = {max(data.shape)}\n Change "sample_size"')
    
    freqs, psd = signal.welch(data, sample_rate, nperseg=win)
    
    
    
    # Plot the power spectrum
    plt.figure(figsize=(8,4))
    for i in range(min(data.shape)):
        plt.plot(freqs, psd[i,:], lw=1)
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (V^2 / Hz)')
    plt.ylim((0.001, psd.max() * 1.1))
    plt.title(title)
    plt.xlim([0, freqs.max()])
    plt.legend(chan_names[:min(data.shape)])
    plt.show()



##############################################################################
#                               Main                                         #
##############################################################################


if __name__ == '__main__':
    
    # Changing the working directory to the one where this file is located.
    fpath = os.path.dirname(os.path.realpath(__file__))
    if fpath != os.getcwd():
        os.chdir(fpath)
    
    # Parameters
    # fname      = 'OpenBCI-RAW-2022-01-06_17-10-09.txt'       # dummy-data
    fname      = 'OpenBCI-RAW-2022-01-10_16-53-01.txt'       # mk1 monday
    
    
    lowcut     = 0.5 # [Hz] Lower cutoff frequencies for bandpass filter
    highcut    = 60  # [Hz] Higher cutoff frequencies for bandpass filter
    
    chan_names = ['Ch 0', 'Ch 1', 'Ch 2', 'Ch 3', 'Ch 4', 'Ch 5', 'Ch 6',
                  'Ch 7', 'Ch 8', 'Ch 9', 'Ch 10', 'Ch 11', 'Ch 12', 'Ch 13',
                  'Ch 14', 'Ch 15'] # Names of the channels
    
    
    
    # Load the openBCI data file into the script
    nb_chans, sample_rate, raw_data = load_openBCI_data(fname)
    
    
    
    # 2D np array of the EEG data
    X = raw_data.filter(regex='EXG Channel').to_numpy()

    
    # Removing the samples which arent finished yet.
    sample_size          = sample_rate * 4 # 1 sample is a 3 sec long time series
    total_nb_measurement = X.shape[0]      # The total amount of measurement pr. channel
    unfinished_samples   = int(total_nb_measurement % sample_size)
    X                    = X[:-unfinished_samples,:]
    
    # Turning the np array into a 3D array, and transposing it so the channels 
    # become the rows, instead of the columns.
    X = X.reshape(int(X.shape[0]/sample_size), sample_size, nb_chans).transpose(0,2,1)
    
    
    # Plot the nb_chans channels in 1 plot.
    plot_eeg(X[0,:,:], channels = nb_chans, vspace = 5000, samplerate = sample_rate)
    
    
    # Plot the Power FFT power distribution (Welch's periodogram)
    plot_FFT(X[0,:,:], sample_rate = sample_rate, lowest_freq = lowcut,
              title = 'Pre-filtered FFT')
    
    
    
    # Filtering the data with a notch filter @ 50 and 25 hz
    X[0,:,:] = notch_filter(X[0,:,:], sample_rate = sample_rate,
                              notch_freqs = [50,25], show_plot = True)
    





###############################################################################


###############################################################################
    
    """
    
    # Bandpass filter from 0.5Hz to 50 Hz (The cutoff rate must be half as big
    # as the sample rate, to fulfill Nyquist theorem)
    y = butter_bandpass_filter(X, lowcut, highcut, sample_rate)
    
    
    
    

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = sample_rate
    

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = sosfreqz(sos, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')

    # Filter a noisy signal.
    T = 0.05
    nsamples = 50 #T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.show()
    
    
    
    # Notch filter @ 50Hz
    
    """
    
    
   
    
