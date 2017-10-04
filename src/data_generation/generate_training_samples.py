# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import sys
import time
import argparse
import pprint
import json

import numpy as np
import pandas as pd
import h5py

from matplotlib import mlab
from scipy.interpolate import interp1d
from librosa.feature import melspectrogram
from librosa import logamplitude
from scipy.signal import butter, filtfilt

from IPython import embed


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


def get_psd(real_strain, sampling_rate=4096):

    # Define some constants
    nfft = 2 * sampling_rate  # Bigger values yield better resolution?

    # Use matplotlib.mlab to calculate the PSD from the real strain
    P_xx, freqs = mlab.psd(real_strain, NFFT=nfft, Fs=sampling_rate)

    # Interpolate it linearly, so we can re-sample the spectrum arbitrarily
    psd = interp1d(freqs, P_xx)

    return psd


def get_waveforms_as_dataframe(waveforms_path):

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


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class CustomArgumentParser:

    def __init__(self):

        # Set up the parser
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
        self.parser = argparse.ArgumentParser(formatter_class=formatter_class)

        # Add command line options
        self.parser.add_argument('--n-samples',
                                 help='Number of samples to generate',
                                 type=int,
                                 default=64)
        self.parser.add_argument('--sample-length',
                                 help='Sample length in seconds',
                                 type=int,
                                 default=12)
        self.parser.add_argument('--sampling-rate',
                                 help='Sampling rate in Hz',
                                 type=int,
                                 default=4096)
        self.parser.add_argument('--n-injections',
                                 help='Number of injections per sample',
                                 type=int,
                                 default=2)
        self.parser.add_argument('--loudness',
                                 help='Scaling factor for injections',
                                 type=float,
                                 default=1.0)
        self.parser.add_argument('--data-path',
                                 help='Path of the data directory',
                                 type=str,
                                 default='../data/')
        self.parser.add_argument('--waveforms-file',
                                 help='Name of the file containing the '
                                      'pre-computed waveforms',
                                 type=str,
                                 default='samples_dist_100_300.h5')
        self.parser.add_argument('--output-file',
                                 help='Name of the ouput HDF file',
                                 type=str,
                                 default='training_samples.h5')
        self.parser.add_argument('--noise-type',
                                 help='Type of noise used for injections',
                                 choices=['gaussian', 'real'],
                                 default='real')

    # -------------------------------------------------------------------------

    def parse_args(self):

        # Parse arguments and return them as a dict instead of Namespace
        return self.parser.parse_args().__dict__

    # -------------------------------------------------------------------------


class Spectrogram:

    def __init__(self, sample_length, sampling_rate, n_injections,
                 waveforms, real_strains, psds, noise_type, max_delta_t=0.0,
                 loudness=1.0):

        # Store all parameters passed as arguments
        self.sample_length = sample_length
        self.sampling_rate = sampling_rate
        self.n_injections = n_injections
        self.waveforms = waveforms
        self.real_strains = real_strains
        self.psds = psds
        self.noise_type = noise_type
        self.max_delta_t = max_delta_t
        self.loudness = loudness

        # Initialize all other class attributes
        self.chirpmasses = None
        self.distances = None
        self.injections = None
        self.labels = None
        self.noises = None
        self.pad = 3
        self.positions = None
        self.signals = None
        self.spectrograms = None
        self.strains = None

        # Add padding so that we can remove the fringe effects due to applying
        # the PSD and calculating the spectrogram later on
        self.length = (self.sample_length + 2 * self.pad) * self.sampling_rate

        # Create a random time difference between the signals for H1 and L1
        self.delta_t = np.random.uniform(-1 * max_delta_t, max_delta_t)
        self.offset = int(self.delta_t * self.sampling_rate)

        # Create the noises (self.noise_type determines with we use Gaussian
        # noise or real noise from the detector recordings)
        self.noises = self._make_noises()

        # Create the signals by making some injections
        self.signals, self.injections = self._make_signals()

        # Calculate the strains as the sum of the noises and the signals
        self.strains = self._make_strains()

        # Calculate a spectrogram from the strain
        self.spectrograms = self._make_spectrograms()

        # Finally calculate the labels for this spectrogram:
        # - label: Injection present or not? (0 or 1)
        # - chirpmass: Chirp mass parameter of the injection
        # - distance: Distance parameter of the injection
        self.labels, self.chirpmasses, self.distances = self._make_labels()

        # Finally, remove the padding again
        self.spectrograms, self.labels, self.chirpmasses, self.distances = \
            self._remove_padding()

    # -------------------------------------------------------------------------

    def _make_noises(self):

        noises = dict()

        # In case we are working with simulated noise, just create some
        # Gaussian noise with the correct length
        if self.noise_type == 'gaussian':

            noises['H1'] = np.random.normal(0, 1, self.length)
            noises['L1'] = np.random.normal(0, 1, self.length)

        # If we are using real noise, select some random subset of the
        # provided noise, i.e. a random piece of the real strain data
        elif self.noise_type == 'real':

            # Find maximum starting position
            max_pos = dict()
            max_pos['H1'] = len(real_strains['H1']) - self.length
            max_pos['L1'] = len(real_strains['L1']) - self.length

            # Randomly find the starting positions
            start = dict()
            start['H1'] = int(np.random.uniform(0, max_pos['H1']))
            start['L1'] = int(np.random.uniform(0, max_pos['L1']))

            # Find the end positions
            end = dict()
            end['H1'] = int(start['H1'] + self.length)
            end['L1'] = int(start['L1'] + self.length)

            # Select random chunks of the real detector recording as the noise
            noises['H1'] = self.real_strains['H1'][start['H1']:end['H1']]
            noises['L1'] = self.real_strains['L1'][start['L1']:end['L1']]

        return noises

    # -------------------------------------------------------------------------

    def _make_signals(self):

        # Initialize empty signals
        signals = dict()
        signals['H1'] = np.zeros(self.length)
        signals['L1'] = np.zeros(self.length)

        # Get the length of a single waveform
        waveform_length = (len(self.waveforms.iloc[0]['waveform']) /
                           self.sampling_rate)

        # Calculate the start positions of the injections
        spacing = self.sample_length - self.n_injections * waveform_length
        spacing = spacing / (self.n_injections + 1)
        self.positions = [(self.pad + spacing + i * (waveform_length +
                           spacing)) * self.sampling_rate for i in
                          range(self.n_injections)]

        # Initialize an empty list for the injection meta-information
        injections = []

        # Loop over all injections to be made
        for inj_number in range(self.n_injections):

            # Randomly select a row from the waveforms DataFrame
            waveform_idx = np.random.randint(len(self.waveforms))
            waveform_row = self.waveforms.iloc[waveform_idx]

            # Extract the waveform, chirpmass and distance from that row
            waveform = dict()
            waveform['H1'] = waveform_row['waveform']
            waveform['L1'] = waveform_row['waveform']
            chirpmass = waveform_row['chirpmass']
            distance = waveform_row['distance']

            # Now we randomly chop off between 0 and 2 seconds from the
            # start, to make the signals variable in length
            cut_off = np.random.uniform(0, 2)
            waveform['H1'] = waveform['H1'][int(cut_off * self.sampling_rate):]
            waveform['L1'] = waveform['L1'][int(cut_off * self.sampling_rate):]

            # If we are using simulated Gaussian noise, we have to apply the
            # PSD directly to the waveform(s)
            if self.noise_type == 'gaussian':

                # Apply the Power Spectral Density to create a colored signal
                waveform['H1'] = apply_psd(waveform['H1'], self.psds['H1'])
                waveform['L1'] = apply_psd(waveform['L1'], self.psds['L1'])

                # Cut off spectral leakage that is due to the Fourier Transform
                waveform['H1'] = waveform['H1'][100:-50]
                waveform['L1'] = waveform['L1'][100:-50]

            # Calculate absolute starting positions of the injections
            abs_start_pos = dict()
            abs_start_pos['H1'] = self.positions[inj_number]
            abs_start_pos['L1'] = self.positions[inj_number] + self.offset

            # Calculate absolute waveform lengths of the injections
            abs_waveform_length = dict()
            abs_waveform_length['H1'] = len(waveform['H1'])
            abs_waveform_length['L1'] = len(waveform['L1'])

            # Calculate relative starting positions of the injections
            rel_start_pos = dict()
            rel_start_pos['H1'] = abs_start_pos['H1'] / len(signals['H1'])
            rel_start_pos['L1'] = abs_start_pos['L1'] / len(signals['H1'])

            # Calculate relative waveform lengths of the injections
            rel_waveform_length = dict()
            rel_waveform_length['H1'] = abs_waveform_length['H1'] / self.length
            rel_waveform_length['L1'] = abs_waveform_length['L1'] / self.length

            # Calculate the absolute end position of the injection
            abs_end_pos = dict()
            abs_end_pos['H1'] = abs_start_pos['H1'] + abs_waveform_length['H1']
            abs_end_pos['L1'] = abs_start_pos['L1'] + abs_waveform_length['L1']

            # Make the injection, i.e. add the waveform to the signal
            signals['H1'][int(abs_start_pos['H1']):int(abs_end_pos['H1'])] += \
                self.loudness * waveform['H1']
            signals['L1'][int(abs_start_pos['L1']):int(abs_end_pos['L1'])] += \
                self.loudness * waveform['L1']

            # Store information about the injection we just made
            injections.append(dict(waveform_idx=waveform_idx,
                                   chirpmass=chirpmass,
                                   cut_off=cut_off,
                                   distance=distance,
                                   rel_start_pos=rel_start_pos,
                                   rel_waveform_length=rel_waveform_length))

        return signals, injections

    # -------------------------------------------------------------------------

    def _make_strains(self):

        strains = dict()
        strains['H1'] = self.noises['H1'] + self.signals['H1']
        strains['L1'] = self.noises['L1'] + self.signals['L1']

        # If we are using real noise, we have to apply the PSD to the sum of
        # noise and signal to 'whiten' the strain:
        if self.noise_type == 'real':

            # Apply the Power Spectral Density to whiten
            strains['H1'] = apply_psd(strains['H1'], self.psds['H1'])
            strains['L1'] = apply_psd(strains['L1'], self.psds['L1'])

        return strains

    # -------------------------------------------------------------------------

    def _make_spectrograms(self):

        # Essentially curry the melspectrogram() function of librosa, because
        # we need to call it twice and this is just more readable
        def make_spectrogram(strain):
            return melspectrogram(strain, sr=4096, n_fft=1024, hop_length=64,
                                  n_mels=64, fmin=1, fmax=600)

        # Calculate the pure spectrograms
        spectrograms = dict()
        spectrograms['H1'] = make_spectrogram(self.strains['H1'])
        spectrograms['L1'] = make_spectrogram(self.strains['L1'])

        # Make the spectrograms log-amplitude
        spectrograms['H1'] = logamplitude(spectrograms['H1'])
        spectrograms['L1'] = logamplitude(spectrograms['L1'])

        return spectrograms

    # -------------------------------------------------------------------------

    def _make_labels(self):

        # Get the lengths of the spectrograms
        lengths = dict()
        lengths['H1'] = self.spectrograms['H1'].shape[1]
        lengths['L1'] = self.spectrograms['L1'].shape[1]

        # Initialize an empty vector for the label (injection yes / no?)
        labels = dict()
        labels['H1'] = np.zeros(lengths['H1'])
        labels['L1'] = np.zeros(lengths['L1'])

        # Initialize an empty vector for the chirpmass
        chirpmasses = dict()
        chirpmasses['H1'] = np.zeros(lengths['H1'])
        chirpmasses['L1'] = np.zeros(lengths['L1'])

        # Initialize an empty vector for the distance
        distances = dict()
        distances['H1'] = np.zeros(lengths['H1'])
        distances['L1'] = np.zeros(lengths['L1'])

        # Loop over the injections we made and add set the label to 1 at
        # every point where there should be an injection present
        for injection in self.injections:

            # Shortcut for the chirpmass and distance of this injection
            chirpmass = injection['chirpmass']
            distance = injection['distance']

            # Calculate start positions
            start = dict()
            start['H1'] = injection['rel_start_pos']['H1'] * lengths['H1']
            start['L1'] = injection['rel_start_pos']['L1'] * lengths['L1']

            # Calculate end positions
            end = dict()
            end['H1'] = start['H1'] + (injection['rel_waveform_length']['H1'] *
                                       lengths['H1'])
            end['L1'] = start['L1'] + (injection['rel_waveform_length']['L1'] *
                                       lengths['L1'])

            # Set the labels vector
            labels['H1'][int(start['H1']):int(end['H1'])] = 1
            labels['L1'][int(start['L1']):int(end['L1'])] = 1

            # Set the chirpmasses vector
            chirpmasses['H1'][int(start['H1']):int(end['H1'])] = chirpmass
            chirpmasses['L1'][int(start['L1']):int(end['L1'])] = chirpmass

            # Set the distances vector
            distances['H1'][int(start['H1']):int(end['H1'])] = distance
            distances['L1'][int(start['L1']):int(end['L1'])] = distance

        return labels, chirpmasses, distances

    # -------------------------------------------------------------------------

    def _remove_padding(self):

        # Get the lengths of the spectrograms
        lengths = dict()
        lengths['H1'] = self.spectrograms['H1'].shape[1]
        lengths['L1'] = self.spectrograms['L1'].shape[1]

        # Get the start of the "inner part" with the injections
        start = dict()
        start['H1'] = int((lengths['H1'] / self.length) * self.sampling_rate
                          * self.pad)
        start['L1'] = int((lengths['H1'] / self.length) * self.sampling_rate
                          * self.pad)

        # Get the end of the "inner part" with the injections
        end = dict()
        end['H1'] = -start['H1']
        end['L1'] = -start['L1']

        # For the spectrograms, only select the "inner part"
        spectrograms = dict()
        spectrograms['H1'] = self.spectrograms['H1'][:, start['H1']:end['H1']]
        spectrograms['L1'] = self.spectrograms['L1'][:, start['L1']:end['L1']]

        # For the label vectors, only select the "inner part"
        labels = dict()
        labels['H1'] = self.labels['H1'][start['H1']:end['H1']]
        labels['L1'] = self.labels['L1'][start['L1']:end['L1']]

        # For the chirpmass vectors, only select the "inner part"
        chirpmasses = dict()
        chirpmasses['H1'] = self.chirpmasses['H1'][start['H1']:end['H1']]
        chirpmasses['L1'] = self.chirpmasses['L1'][start['L1']:end['L1']]

        # For the distances vectors, only select the "inner part"
        distances = dict()
        distances['H1'] = self.distances['H1'][start['H1']:end['H1']]
        distances['L1'] = self.distances['L1'][start['L1']:end['L1']]

        return spectrograms, labels, chirpmasses, distances

    # -------------------------------------------------------------------------

    def get_spectrograms(self):

        # Stack the spectrograms. This produces the NHWC, or "channels last"
        # format, which is the standard for keras (but not for PyTorch!)
        return np.dstack((self.spectrograms['H1'], self.spectrograms['L1']))

    # -------------------------------------------------------------------------

    def get_label(self):

        return np.maximum(self.labels['H1'], self.labels['L1']).astype('int')

    # -------------------------------------------------------------------------

    def get_chirpmass(self):

        return np.maximum(self.chirpmasses['H1'], self.chirpmasses['L1'])

    # -------------------------------------------------------------------------

    def get_distance(self):

        return np.maximum(self.distances['H1'], self.distances['L1'])

    # -------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Set the seed for the random number generator
    np.random.seed(42)

    # Start the stopwatch
    start_time = time.time()
    print('Starting sample generation using the following parameters:')

    # Read in command line options
    parser = CustomArgumentParser()
    arguments = parser.parse_args()
    pprint.pprint(arguments)

    # Shortcuts for some global parameters / file paths
    data_path = arguments['data_path']
    waveforms_file = arguments['waveforms_file']

    # -------------------------------------------------------------------------
    # Read in the real strain data from the LIGO website
    # -------------------------------------------------------------------------

    print('Reading in real strain data for PSD computation...', end=' ')

    # Build the paths to the files with the real strain data
    real_strain_file = dict()
    real_strain_file['H1'] = 'H1_2017_4096.hdf5'
    real_strain_file['L1'] = 'L1_2017_4096.hdf5'
    strain_path_H1 = os.path.join(data_path, 'strain', real_strain_file['H1'])
    strain_path_L1 = os.path.join(data_path, 'strain', real_strain_file['L1'])

    # Read the HDF files into numpy arrays and store them in a dict
    real_strains = dict()
    with h5py.File(strain_path_H1) as file:
        real_strains['H1'] = np.array(file['strain/Strain'])
    with h5py.File(strain_path_L1) as file:
        real_strains['L1'] = np.array(file['strain/Strain'])

    print('Done!')

    # -------------------------------------------------------------------------
    # Pre-calculate the Power Spectral Density from the real strain data
    # -------------------------------------------------------------------------

    print('Computing the PSD of the real strain...', end=' ')
    psds = dict()
    psds['H1'] = get_psd(real_strains['H1'])
    psds['L1'] = get_psd(real_strains['L1'])
    print('Done!')

    # -------------------------------------------------------------------------
    # Load the pre-calculated waveforms from an HDF file
    # -------------------------------------------------------------------------

    print('Loading pre-computed waveforms...', end=' ')
    waveforms_path = os.path.join(data_path, 'waveforms', waveforms_file)
    waveforms = get_waveforms_as_dataframe(waveforms_path)
    print('Done!')

    # -------------------------------------------------------------------------
    # Generate spectrograms
    # -------------------------------------------------------------------------

    # Store away the spectrograms and labels that we create
    spectograms, labels, chirpmasses, distances = [], [], [], []

    print('Generating training samples...')

    # TODO: Can this loop be parallelized?
    for i in range(arguments['n_samples']):

        # Create a spectrogram
        spectogram = Spectrogram(sample_length=arguments['sample_length'],
                                 sampling_rate=arguments['sampling_rate'],
                                 n_injections=arguments['n_injections'],
                                 loudness=arguments['loudness'],
                                 waveforms=waveforms,
                                 psds=psds,
                                 real_strains=real_strains,
                                 noise_type=arguments['noise_type'],
                                 max_delta_t=0.1)

        # Store away the important information (spectrogram and label)
        spectograms.append(spectogram.get_spectrograms())
        labels.append(spectogram.get_label())
        chirpmasses.append(spectogram.get_chirpmass())
        distances.append(spectogram.get_distance())

        # Make a sweet progress bar to see how things are going
        progress_bar(current_value=i+1, max_value=arguments['n_samples'],
                     elapsed_time=time.time() - start_time)

    # -------------------------------------------------------------------------
    # Save the results as and HDF file
    # -------------------------------------------------------------------------

    print('\nSaving results...', end=' ')

    training_path = os.path.join(data_path, 'training',
                                 arguments['output_file'])

    with h5py.File(training_path, 'w') as file:
        file['spectrograms'] = np.array(spectograms)
        file['labels'] = np.array(labels)
        file['chirpmasses'] = np.array(chirpmasses)
        file['distances'] = np.array(distances)

    print('Done!')
    file_size = os.path.getsize(training_path) / 1e6
    print('Full sample size: {:.1f} MB'.format(file_size))
