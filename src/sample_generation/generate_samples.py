# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import time
import pprint

import numpy as np
import h5py

from samplegenerators import CustomArgumentParser, Spectrogram, TimeSeries
from tools import get_psd, get_waveforms_as_dataframe, progress_bar


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Set the seed for the random number generator
    np.random.seed(42)

    # Read in command line options
    print('Starting sample generation using the following parameters:')
    parser = CustomArgumentParser()
    arguments = parser.parse_args()
    pprint.pprint(arguments)

    # Shortcuts for some global parameters / file paths
    data_path = arguments['data_path']
    waveforms_file = arguments['waveforms_file']
    sample_type = arguments['sample_type']
    strain_file = arguments['strain_file']
    use_type = arguments['use_type']

    # -------------------------------------------------------------------------
    # Read in the real strain data from the LIGO website
    # -------------------------------------------------------------------------

    print('Reading in real strain data for PSD computation...', end=' ')

    # Names of the files containing the real strains, i.e. detector recordings
    real_strain_file = {'H1': '{}_H1_STRAIN_4096.hdf5'.format(strain_file),
                        'L1': '{}_L1_STRAIN_4096.hdf5'.format(strain_file)}

    # Read the HDF files into numpy arrays and store them in a dict
    real_strains = dict()
    for ifo in ['H1', 'L1']:

        # Make the full path for the strain file
        strain_path = os.path.join(data_path, 'strain', real_strain_file[ifo])

        # Read the HDF file into a numpy array
        with h5py.File(strain_path) as file:
            real_strains[ifo] = np.array(file['strain/Strain'])

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
    # Load the pre-calculated waveforms from an HDF file into a DataFrame
    # -------------------------------------------------------------------------

    print('Loading pre-computed waveforms...', end=' ')
    waveforms_path = os.path.join(data_path, 'waveforms', waveforms_file)
    waveforms = get_waveforms_as_dataframe(waveforms_path)
    print('Done!')

    # -------------------------------------------------------------------------
    # Generate spectrograms
    # -------------------------------------------------------------------------

    if sample_type == 'timeseries':
        MakeSample = TimeSeries
        method_name = "get_timeseries"
    else:
        MakeSample = Spectrogram
        method_name = "get_spectrograms"

    # Initialize a dict of lists where we store our results
    results = dict()
    results[sample_type] = []
    results['labels'] = []
    results['chirpmasses'] = []
    results['distances'] = []
    results['snrs'] = []

    # Start the stopwatch
    start_time = time.time()
    print('Generating {} samples...'.format(use_type))

    # TODO: Can this loop be parallelized?
    for i in range(arguments['n_samples']):

        # Create a spectrogram or a time series
        sample = MakeSample(sample_length=arguments['sample_length'],
                            sampling_rate=arguments['sampling_rate'],
                            max_n_injections=arguments['max_n_injections'],
                            loudness=arguments['loudness'],
                            noise_type=arguments['noise_type'],
                            pad=arguments['pad'],
                            waveforms=waveforms,
                            psds=psds,
                            real_strains=real_strains,
                            max_delta_t=0.1)

        # Store away Spectrogram / TimeSeries and labels
        results[sample_type].append(getattr(sample, method_name)())
        results['labels'].append(sample.get_label())
        results['chirpmasses'].append(sample.get_chirpmass())
        results['distances'].append(sample.get_distance())
        results['snrs'].append(sample.get_snr())

        # Make a sweet progress bar to see how things are going
        progress_bar(current_value=i+1, max_value=arguments['n_samples'],
                     elapsed_time=time.time()-start_time)

    # -------------------------------------------------------------------------
    # Save the results as an HDF file
    # -------------------------------------------------------------------------

    print('\nSaving results...', end=' ')

    results_path = os.path.join(data_path, use_type, sample_type,
                                arguments['output_file'])

    with h5py.File(results_path, 'w') as file:
        file[sample_type] = np.array(results[sample_type])
        file['labels'] = np.array(results['labels'])
        file['chirpmasses'] = np.array(results['chirpmasses'])
        file['distances'] = np.array(results['distances'])

    print('Done!')
    file_size = os.path.getsize(results_path) / 1e6
    print('Full sample size: {:.1f} MB'.format(file_size))
