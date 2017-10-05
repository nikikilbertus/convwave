# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import abc
import numpy as np
import argparse

from librosa.feature import melspectrogram
from librosa import logamplitude

from tools import apply_psd


# -----------------------------------------------------------------------------
# CLASS FOR PARSING COMMAND LINE ARGUMENTS FOR THE GENERATORS
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
        self.parser.add_argument('--pad',
                                 help='Noise padding to avoid spectral '
                                      'leakage (lengths in seconds)',
                                 type=float,
                                 default=3.0)
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
        self.parser.add_argument('--sample-type',
                                 help='Type of sample to create',
                                 choices=['timeseries', 'spectrograms'],
                                 default='spectrogram')

    # -------------------------------------------------------------------------

    def parse_args(self):

        # Parse arguments and return them as a dict instead of Namespace
        return self.parser.parse_args().__dict__

    # -------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# BASE CLASS FOR THE SPECTROGRAM AND THE TIME SERIES SAMPLE GENERATOR
# -----------------------------------------------------------------------------

class SampleGenerator:

    def __init__(self, sample_length, sampling_rate, n_injections,
                 waveforms, real_strains, psds, noise_type, max_delta_t=0.0,
                 loudness=1.0, pad=3.0):

        # Store all parameters passed as arguments
        self.sample_length = sample_length
        self.sampling_rate = sampling_rate
        self.n_injections = n_injections
        self.waveforms = waveforms
        self.real_strains = real_strains
        self.pad = pad
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
        self.positions = None
        self.signals = None
        self.strains = None

        # Add padding so that we can remove the fringe effects due to applying
        # the PSD and calculating the spectrogram later on
        self.length = (self.sample_length + 2 * self.pad) * self.sampling_rate
        self.length = int(self.length)

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
            max_pos['H1'] = len(self.real_strains['H1']) - self.length
            max_pos['L1'] = len(self.real_strains['L1']) - self.length

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

    def _make_labels(self, lengths):

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

    @abc.abstractmethod
    def _remove_padding(self):
        raise NotImplementedError()

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
# CLASS TO GENERATE SPECTROGRAM SAMPLES
# -----------------------------------------------------------------------------

class Spectrogram(SampleGenerator):

    def __init__(self, sample_length, sampling_rate, n_injections,
                 waveforms, real_strains, psds, noise_type, max_delta_t=0.0,
                 loudness=1.0, pad=3.0):

        # Inherit from the SampleGenerator base class
        super().__init__(sample_length, sampling_rate, n_injections,
                         waveforms, real_strains, psds, noise_type,
                         max_delta_t, loudness, pad)

        # Add a variable for the spectrograms
        self.spectrograms = None

        # Calculate a spectrogram from the strain
        self.spectrograms = self._make_spectrograms()

        # Calculate the labels for this spectrogram
        label_lengths = {'H1': self.spectrograms['H1'].shape[1],
                         'L1': self.spectrograms['L1'].shape[1]}
        self.labels, self.chirpmasses, self.distances = \
            self._make_labels(lengths=label_lengths)

        # Finally, remove the padding again
        self.spectrograms, self.labels, self.chirpmasses, self.distances = \
            self._remove_padding()

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


# -----------------------------------------------------------------------------
# CLASS TO GENERATE TIME SERIES SAMPLES
# -----------------------------------------------------------------------------

class TimeSeries(SampleGenerator):

    def __init__(self, sample_length, sampling_rate, n_injections,
                 waveforms, real_strains, psds, noise_type, max_delta_t=0.0,
                 loudness=1.0, pad=3.0):

        # Inherit from the SampleGenerator base class
        super().__init__(sample_length, sampling_rate, n_injections,
                         waveforms, real_strains, psds, noise_type,
                         max_delta_t, loudness, pad)

        # Calculate the labels for this spectrogram:
        label_lengths = {'H1': len(self.strains['H1']),
                         'L1': len(self.strains['L1'])}
        self.labels, self.chirpmasses, self.distances = \
            self._make_labels(lengths=label_lengths)

        # Remove the padding again
        self.strains, self.labels, self.chirpmasses, self.distances = \
            self._remove_padding()

        # Down-sample to 2048 Hz
        self.strains, self.labels, self.chirpmasses, self.distances = \
            self._downsample()

    # -------------------------------------------------------------------------

    def _remove_padding(self):

        pad_length = int(self.pad * self.sampling_rate)

        # Unpad the strains
        strains = dict()
        strains['H1'] = self.strains['H1'][pad_length:-pad_length]
        strains['L1'] = self.strains['L1'][pad_length:-pad_length]

        # Unpad the labels
        labels = dict()
        labels['H1'] = self.labels['H1'][pad_length:-pad_length]
        labels['L1'] = self.labels['L1'][pad_length:-pad_length]

        # Unpad the chirpmasses
        chirpmasses = dict()
        chirpmasses['H1'] = self.chirpmasses['H1'][pad_length:-pad_length]
        chirpmasses['L1'] = self.chirpmasses['L1'][pad_length:-pad_length]

        # Unpad the distances
        distances = dict()
        distances['H1'] = self.distances['H1'][pad_length:-pad_length]
        distances['L1'] = self.distances['L1'][pad_length:-pad_length]

        return strains, labels, chirpmasses, distances

    # -------------------------------------------------------------------------

    def _downsample(self):
        """
        Downsample the strains and the label, chirpmass and distance vectors
        to half their frequency, i.e. usually from 4096 Hz to 2048 Hz.

        Returns:
            strains: Downsampled version of the strains
            labels: Downsampled version of the label vectors
            chirpmasses: Downsampled version of the chirpmass vectors
            distances: Downsampled version of the distance vectors
        """

        # TODO: Make downsampling more generic, i.e. to arbitrary frequency?
        strains = {'H1': self.strains['H1'][::2],
                   'L1': self.strains['L1'][::2]}
        labels = {'H1': self.labels['H1'][::2],
                  'L1': self.labels['L1'][::2]}
        chirpmasses = {'H1': self.chirpmasses['H1'][::2],
                       'L1': self.chirpmasses['L1'][::2]}
        distances = {'H1': self.distances['H1'][::2],
                     'L1': self.distances['L1'][::2]}

        return strains, labels, chirpmasses, distances

    # -------------------------------------------------------------------------

    def get_timeseries(self):

        return np.dstack((self.strains['H1'], self.strains['L1']))
