# Training

This folder will hold the HDF/H5 files containing training data (i.e. Gaussian noise plus injected waveforms) that can be used to train the network. This README file is necessary so that git keeps the folder in the repository :)

The naming convention for the training files is `training_<dist1>_<dist2>_<n_samples>.h5`, i.e. for example `training_0100_0300_4k.h5` (leading zeros are useful to make sure the samples actually get sorted by distance).