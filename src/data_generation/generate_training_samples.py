# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import sys
import time

import numpy as np
import h5py
import librosa

from matplotlib import mlab
from scipy.interpolate import interp1d


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def progress_bar(current_value, max_value, elapsed_time=0, bar_length=50):
    """
    Print a progress bar to the terminal to see how things are moving along.

    Args:
        current_value: The current number of spectrograms processed.
        max_value: The maximum number of spectrograms to be processed.
        elapsed_time: The time that has elapsed since the start of the script.
        bar_length: Maximum length of the bar.
    """

    # Construct the actual progress bar
    percent = float(current_value) / max_value
    bar = '=' * int(round(percent * bar_length))
    spaces = '-' * (bar_length - len(bar))

    # Calculate the estimated time remaining
    eta = elapsed_time / percent - elapsed_time

    # Collect the outputs and write them to stdout; move the carriage back
    # to the start of the line so that the progress bar is always updated.
    out = ("\r[{0}] {1}% ({2}/{3}) | {4:.1f}s elapsed | ETA: {5:.1f}s".
           format(bar + spaces, int(round(percent * 100)), current_value,
                  max_value, elapsed_time, eta))
    sys.stdout.write(out)
    sys.stdout.flush()


def add_color(signal_t, psd, sampling_rate=4096):
    """
    Take a signal in the time domain, and a precalculated Power Spectral
    Density, and color the signal according to the given PSD.

    Args:
        signal_t: A signal in time domain (i.e. a 1D numpy array)
        psd: A Power Spectral Density, e.g. calculated from the detector noise.
            Should be a function: psd(frequency)
        sampling_rate: Sampling rate of signal_t

    Returns: color_signal_t, the colored signal in the time domain.
    """

    # First set some parameters for computing power spectra
    n = len(signal_t)
    dt = 1./sampling_rate

    # Go into Fourier (frequency) space: signal_t -> signal_f
    frequencies = np.fft.rfftfreq(n, dt)
    signal_f = np.fft.rfft(signal_t)

    # Divide by the given Power Spectral Density (PSD)
    # This is the 'whitening' = actually adding color
    color_signal_f = signal_f / (np.sqrt(psd(frequencies) / dt / 2.))

    # Go back into time space: color_signal_f -> color_signal_t
    color_signal_t = np.fft.irfft(color_signal_f, n=n)

    # In case we want to use a Butterworth-filter, here's how to do it:
    # bb, ab = butter(4, [20*2/4096, 300*2/4096], btype="band")
    # color_signal_t = filtfilt(bb, ab, color_signal_t)

    return color_signal_t


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class Spectrogram:

    def __init__(self, sample_length, sampling_rate, n_injections,
                 waveforms, psd, loudness=1.0):

        self.sample_length = sample_length
        self.sampling_rate = sampling_rate
        self.n_injections = n_injections
        self.psd = psd
        self.waveforms = waveforms
        self.positions = None
        self.loudness = loudness

        # Create the Gaussian noise
        self.noise = self._make_noise()

        # Create the signal by making some injections
        self.signal, self.injections = self._make_signal()

        # Calculate the strain as the sum of the noise and the signal
        self.strain = self._make_strain()

        # Calculate a spectrogram from the strain
        self.spectrogram = self._make_spectrogram()

        # Finally calculate the label for this spectrogram
        self.label = self._make_label()

    # -------------------------------------------------------------------------

    def _make_noise(self):
        return np.random.normal(0, 1, self.sample_length * self.sampling_rate)

    # -------------------------------------------------------------------------

    def _make_signal(self):

        # Initialize an empty signal
        signal = np.zeros(self.sample_length * self.sampling_rate)

        # Get the starting positions for each injection
        self.positions = np.linspace(0, 0.9*len(signal), self.n_injections + 1)
        self.positions = np.array([int((x+y)/2) for x, y in
                                   zip(self.positions, self.positions[1:])])

        # Initialize an empty list for the injection meta-information
        injections = []

        # Loop over all injections to be made
        for inj_number in range(self.n_injections):

            # Randomly select a waveform
            waveform_idx = np.random.randint(len(self.waveforms))
            waveform = self.waveforms[waveform_idx]

            # Add color to that waveform
            color_waveform = add_color(waveform, self.psd)

            # Calculate absolute and relative starting positions and lengths
            # of the waveform that is being injected
            abs_start_pos = self.positions[inj_number]
            rel_start_pos = abs_start_pos / len(signal)
            abs_waveform_length = len(color_waveform)
            rel_waveform_length = (abs_waveform_length / len(signal))

            # Calculate the absolute end position of the injection
            abs_end_pos = abs_start_pos + abs_waveform_length

            # Make the injection, i.e. add the waveform to the signal
            signal[abs_start_pos:abs_end_pos] += (self.loudness *
                                                  color_waveform)

            # Store information about the injection we just made
            injections.append(dict(waveform_idx=waveform_idx,
                                   rel_start_pos=rel_start_pos,
                                   rel_waveform_length=rel_waveform_length))

        return signal, injections

    # -------------------------------------------------------------------------

    def _make_strain(self):
        return self.noise + self.signal

    # -------------------------------------------------------------------------

    def _make_spectrogram(self, log_scale=True):

        spectrogram = librosa.feature.melspectrogram(self.strain,
                                                     sr=4096,
                                                     n_fft=1024,
                                                     hop_length=64,
                                                     n_mels=64,
                                                     fmin=0,
                                                     fmax=400)
        log_spectrogram = librosa.logamplitude(spectrogram, ref=1.0)

        if log_scale:
            return log_spectrogram
        return spectrogram

    # -------------------------------------------------------------------------

    def _make_label(self):

        # Get the length of the spectrogram
        spectrogram_length = self.spectrogram.shape[1]

        # Initialize and empty label
        label = np.zeros(spectrogram_length)

        # Loop over the injections we made and add set the label to 1 at
        # every point where there should be an injection present
        for injection in self.injections:
            start = int(injection['rel_start_pos'] * spectrogram_length)
            end = start + int(injection['rel_waveform_length'] *
                              spectrogram_length)
            label[start:end] += 1

        return label

    # -------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Start the stopwatch
    start_time = time.time()
    print('Starting training sample generation...')

    # Path to the directory where all data is stored
    data_path = '../data/'

    # -------------------------------------------------------------------------
    # Read in the real strain data from the LIGO website
    # -------------------------------------------------------------------------

    print('Reading in real strain data for PSD computation...', end=' ')
    with h5py.File(os.path.join(data_path, 'H1_2017_4096.hdf5')) as file:
        real_strain = np.array(file['strain/Strain'])
    print('Done!')

    # -------------------------------------------------------------------------
    # Pre-calculate the Power Spectral Density from the real strain data
    # -------------------------------------------------------------------------

    print('Computing the PSD of the real strain...', end=' ')

    # Define some constants
    sampling_rate = 4096     # The sampling rate for the Fourier transform?
    nfft = 8192              # Bigger values yield better resolution?
    dt = 1. / sampling_rate  # Inverse sampling rate

    # Use matplotlib.mlab to calculate the PSD from the real strain
    P_xx, freqs = mlab.psd(real_strain, NFFT=nfft, Fs=sampling_rate)

    # Interpolate it linearly, so we can re-sample the spectrum at other points
    psd = interp1d(freqs, P_xx)

    print('Done!')

    # -------------------------------------------------------------------------
    # Load the pre-calculated waveforms from an HDF file
    # -------------------------------------------------------------------------

    print('Loading pre-computed waveforms...', end=' ')
    with h5py.File(os.path.join(data_path, 'waveforms_near.h5')) as file:

        # Read in waveforms as well as the indices of the unusable waveforms
        waveforms = np.array(file['waveforms'])
        failed_idx = np.array(file['failed'])

        # Exclude failed waveforms right away:
        waveforms = np.array([_ for i, _ in enumerate(waveforms)
                              if i not in failed_idx])
    print('Done!')

    # -------------------------------------------------------------------------
    # Generate spectrograms
    # -------------------------------------------------------------------------

    # Define some global functions
    n_samples = 128
    sample_length = 10
    sampling_rate = 4096
    n_injections = 4
    loudness = 1

    # Store away the spectrograms and labels that we create
    spectograms, labels = [], []

    print('Generating training samples...')

    for i in range(n_samples):

        # Create a spectrogram
        spectogram = Spectrogram(sample_length, sampling_rate, n_injections,
                                 waveforms, psd, loudness)

        # Store away the important information (spectrogram and label)
        spectograms.append(spectogram.spectrogram)
        labels.append(spectogram.label)

        # Make a sweet progress bar to see how things are going
        progress_bar(current_value=i+1, max_value=n_samples,
                     elapsed_time=time.time() - start_time)

    # -------------------------------------------------------------------------
    # Save the results as and HDF file
    # -------------------------------------------------------------------------

    print('\nSaving results...', end=' ')

    filename = os.path.join(data_path, 'training_samples.h5')
    with h5py.File(filename, 'w') as file:

        file['spectrograms'] = np.array(spectograms)
        file['labels'] = np.array(labels)

    print('Done!')
    file_size = os.path.getsize(filename) / 1e6
    print('Full sample size: {:.1f} MB'.format(file_size))
