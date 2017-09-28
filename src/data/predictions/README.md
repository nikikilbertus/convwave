# Predictions

This folder will hold the HDF/H5 files containing the predictions on the test set made by the Fully Convolutional Network (i.e. usually `pytorch_train.py`), and this README file is necessary so that git keeps the folder in the repository :)

The naming convention for the prediction files is `predictions_<dist1>_<dist2>_<n_samples>.h5`, i.e. for example `predictions_0250_0500_1k.h5` (leading zeros are useful to make sure the files actually get sorted by distance).