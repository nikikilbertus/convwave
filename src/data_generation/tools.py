# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import sys
import json

import numpy as np
import pandas as pd
import h5py

from scipy.signal import butter, filtfilt
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


# -----------------------------------------------------------------------------


def apply_psd(signal_t, psd, sampling_rate=4096, apply_butter=False):
    """
    Take a signal in the time domain, and a precalculated Power Spectral
    Density, and color the signal according to the given PSD.

    Args:
        signal_t: A signal in time domain (i.e. a 1D numpy array)
        psd: A Power Spectral Density, e.g. calculated from the detector noise.
            Should be a function: psd(frequency)
        sampling_rate: Sampling rate of signal_t
        apply_butter: Whether or not to apply a Butterworth filter to the data.

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
    if apply_butter:
        f_low = 1
        f_high = 600
        bb, ab = butter(4, [f_low*2/4096, f_high*2/4096], btype="bandpass")
        normalization = np.sqrt((f_high - f_low) / (sampling_rate / 2))
        color_signal_t = filtfilt(bb, ab, color_signal_t) / normalization

    return color_signal_t


# -----------------------------------------------------------------------------


def get_psd(real_strain, sampling_rate=4096):
    """
    Take a detector recording and calculate the Power Spectral Density (PSD).

    Args:
        real_strain: The detector recording to be used.
        sampling_rate: The sampling rate (in Hz) of the recording

    Returns:
        psd: The Power Spectral Density of the detector recordings
    """

    # Define some constants
    nfft = 2 * sampling_rate  # Bigger values yield better resolution?

    # Use matplotlib.mlab to calculate the PSD from the real strain
    P_xx, freqs = mlab.psd(real_strain, NFFT=nfft, Fs=sampling_rate)

    # Interpolate it linearly, so we can re-sample the spectrum arbitrarily
    psd = interp1d(freqs, P_xx)

    return psd


# -----------------------------------------------------------------------------


def get_waveforms_as_dataframe(waveforms_path):
    """
    Take an HDF file containing pre-generated waveforms (as by the
    waveform_generator.py in this repository) and extract the relevant
    information (waveform, mass 1, mass 2, chirpmass, distance) into a
    pandas DataFrame for convenient access.

    Args:
        waveforms_path: The path to the HDF file containing the waveforms

    Returns:
        df: A pandas DataFrame containing all valid waveforms and their
            corresponding masses, chirpmasses and distances.
    """

    # Shorthand definition of the chirp mass
    def chirp_mass(mass1, mass2):
        return (mass1 * mass2) ** (3 / 5) / (mass1 + mass2) ** (1 / 5)

    # Read in the actual waveforms, the config string (and parse from JSON),
    # and the indices of the failed waveforms
    with h5py.File(waveforms_path, 'r') as file:
        waveforms = np.array(file['waveforms'])
        config = json.loads(file['config'].value.astype('str'))['injections']
        failed_idx = np.array(file['failed'])

    # Create a Pandas DataFrame containing only the relevant columns from the
    # config string (other columns are all trivial at this point)
    columns = ['distance', 'mass1', 'mass2']
    df = pd.DataFrame(config, columns=columns)

    # Add columns for the actual waveforms and the chirp masses
    df['waveform'] = list(waveforms)
    df['chirpmass'] = df.apply(lambda row: chirp_mass(row.mass1, row.mass2),
                               axis=1)

    # Drop the rows with the failed waveforms, and reset the index
    # noinspection PyUnresolvedReferences
    df = df.drop(list(failed_idx)).reset_index(drop=True)

    # Resort columns to order them alphabetically
    df = df[sorted(df.columns)]

    # Return the final DataFrame
    return df
