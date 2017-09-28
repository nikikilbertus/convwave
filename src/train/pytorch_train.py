# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import h5py
import os
import sys
import time
import datetime

from IPython import embed
from tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def create_weights(label, start_size=4, end_size=2):
    """
    Create the weights ('grayzones') for a given label.

    Args:
        label: A vector labeling the pixels that contain an injection
        start_size: Number of pixels to ignore at the start of an injection
        end_size: Number of pixels to ignore at the end of an injections

    Returns: A vector that is 0 for the pixels that should be ignored and 1
        for all other pixels.
    """

    a = np.logical_xor(label, np.roll(label, 1))
    b = np.cumsum(a) % 2

    if start_size == 0:
        c = np.zeros(label.shape)
    else:
        c = np.convolve(a * b, np.hstack((np.zeros(start_size - 1),
                                          np.ones(start_size))),
                        mode="same")

    if end_size == 0:
        d = np.zeros(label.shape)
    else:
        d = np.convolve(a * np.logical_not(b),
                        np.hstack((np.ones(end_size), np.zeros(end_size - 1))),
                        mode="same")

    return np.logical_not(np.logical_or(c, d)).astype('int')


def hamming_dist(y_true, y_pred):
    """
    Calculate the Hamming distance between a given predicted label and the
    true label.

    Args:
        y_true: The true label
        y_pred: The predicted label

    Returns: The Hamming distance between the two vectors
    """

    return np.mean(np.abs(y_true - y_pred), axis=(1, 0))


def progress_bar(current_value, max_value, start_time, **kwargs):
    """
    Print the progress bar during training that contains all relevant
    information, i.e. number of epochs, percentage of processed mini-batches,
    elapsed time, estimated time remaining, as well as all metrics provided.

    Args:
        current_value: Current number of processed mini-batches
        max_value: Number of total mini-batches
        start_time: Absolute timestamp of the moment the epoch began
        **kwargs: Various metrics, e.g. the loss or Hamming distance
    """

    # Some preliminary definitions
    bar_length = 20
    elapsed_time = time.time() - start_time

    # Construct the actual progress bar
    percent = float(current_value) / max_value
    bar = '=' * int(round(percent * bar_length))
    spaces = '-' * (bar_length - len(bar))

    # Calculate the estimated time remaining
    eta = elapsed_time / percent - elapsed_time

    # Start with the default info: Progress Bar, number of processed
    # mini-batches, time elapsed, and estimated time remaining (the '\r' at
    # the start moves the carriage back the start of the line, meaning that
    # the progress bar will be overwritten / updated!)
    out = ("\r[{0}] {1:>3}% ({2:>2}/{3}) | {4:.1f}s elapsed | "
           "ETA: {5:.1f}s | ".format(bar + spaces, int(round(percent * 100)),
                                     int(current_value), int(max_value),
                                     elapsed_time, eta))

    # Add all provided metrics, e.g. loss and Hamming distance
    metrics = []
    for metric, value in sorted(kwargs.items()):
        metrics.append("{}: {:.3f}".format(metric, value))
    out += ' - '.join(metrics) + ' '

    # Actually write the finished progress bar to the command line
    sys.stdout.write(out)
    sys.stdout.flush()


def load_data_as_tensor_datasets(file_path, split_ratios=(0.7, 0.2, 0.1),
                                 shuffle_data=False, random_seed=42):
    """
    Take an HDF file with data (Gaussian Noise with waveform injections) and
    read it in, split it into training, test and validation data, and convert
    it to PyTorch TensorDatasets, which can be used in PyTorch DataLoaders,
    which are in turn useful for looping over the data in mini-batches.

    Args:
        file_path: The path to the HDF file containing the samples.
        split_ratios: The ratio of training:test:validation. This ought to
            sum up to 1!
        shuffle_data: Whether or not to shuffle the data before splitting.
        random_seed: Seed for the random number generator.

    Returns: Spectrograms and their respective labels, combined in a PyTorch
        TensorDataset, for training, test and validation.
    """

    # TODO: We might also want to pre-process (normalize) the data?

    # Set the seed for the random number generator
    np.random.seed(random_seed)

    # Read in the spectrograms from the HDF file
    with h5py.File(file_path, 'r') as file:

        x = np.array(file['spectrograms'])
        y = np.array(file['labels'])

    # Swap axes around to get to NCHW format
    x = np.swapaxes(x, 1, 3)
    x = np.swapaxes(x, 2, 3)

    # Generate the indices for training, test and validation
    idx = np.arange(len(x))

    # Shuffle the indices (data) if requested
    if shuffle_data:
        idx = np.random.permutation(idx)

    # Get the indices for training, test and validation
    splits = np.cumsum(split_ratios)
    idx_train = idx[:int(splits[0]*len(x))]
    idx_test = idx[int(splits[0]*len(x)):int(splits[1]*len(x))]
    idx_validation = idx[int(splits[1]*len(x)):]

    # Select the actual data using these indices
    x_train, y_train = x[idx_train], y[idx_train]
    x_test, y_test = x[idx_test], y[idx_test]
    x_validation, y_validation = x[idx_validation], y[idx_validation]

    # Convert the training and test data to PyTorch / CUDA tensors
    x_train = torch.from_numpy(x_train).float().cuda()
    y_train = torch.from_numpy(y_train).float().cuda()
    x_test = torch.from_numpy(x_test).float().cuda()
    y_test = torch.from_numpy(y_test).float().cuda()
    x_validation = torch.from_numpy(x_validation).float().cuda()
    y_validation = torch.from_numpy(y_validation).float().cuda()

    # Create TensorDatasets for training, test and validation
    tensor_dataset_train = TensorDataset(x_train, y_train)
    tensor_dataset_test = TensorDataset(x_test, y_test)
    tensor_dataset_validation = TensorDataset(x_validation, y_validation)

    # Return the resulting TensorDatasets
    return tensor_dataset_train, tensor_dataset_test, tensor_dataset_validation


# -----------------------------------------------------------------------------
# DEFINE THE MODEL FOR THE FULLY CONVOLUTIONAL NET (FCN)
# -----------------------------------------------------------------------------

class Net(nn.Module):

    # -------------------------------------------------------------------------
    # Initialize the net and define functions for the layers
    # -------------------------------------------------------------------------

    def __init__(self):

        # Inherit from the PyTorch neural net module
        super(Net, self).__init__()

        # Convolutional layers: (in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(2, 128, (3, 7), padding=(1, 3), stride=1)
        self.conv2 = nn.Conv2d(128, 128, (3, 7), padding=(1, 3), stride=1)
        self.conv3 = nn.Conv2d(128, 128, (3, 7), padding=(1, 3), stride=1)
        self.conv4 = nn.Conv2d(128, 128, (3, 7), padding=(1, 3), stride=1)
        self.conv5 = nn.Conv2d(128, 128, (3, 7), padding=(1, 3), stride=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 7), padding=(1, 3), stride=1)
        self.conv7 = nn.Conv2d(128, 128, (3, 7), padding=(1, 3), stride=1)
        self.conv8 = nn.Conv2d(128, 1, (1, 1), padding=0, stride=1)

        # Batch norm layers
        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2 = nn.BatchNorm2d(num_features=128)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)
        self.batchnorm4 = nn.BatchNorm2d(num_features=128)
        self.batchnorm5 = nn.BatchNorm2d(num_features=128)
        self.batchnorm6 = nn.BatchNorm2d(num_features=128)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Padding layers:
        self.pad = nn.ReflectionPad2d((3, 3, 1, 1))

    # -------------------------------------------------------------------------
    # Define a forward pass through the network (apply the layers)
    # -------------------------------------------------------------------------

    def forward(self, x):

        # Layer 1
        # ---------------------------------------------------------------------
        x = self.conv1(x)
        x = func.elu(x)

        # Layers 2 to 3
        # ---------------------------------------------------------------------
        convolutions = [self.conv2, self.conv3, self.conv4, self.conv5,
                        self.conv6, self.conv7]
        batchnorms = [self.batchnorm1, self.batchnorm2, self.batchnorm3,
                      self.batchnorm4, self.batchnorm5, self.batchnorm6]

        for conv, batchnorm in zip(convolutions, batchnorms):
            x = conv(x)
            x = batchnorm(x)
            x = func.elu(x)
            x = self.pool(x)
            x = func.dropout(x, p=0.3)

        # Layer 8
        # ---------------------------------------------------------------------
        x = self.conv8(x)
        x = func.sigmoid(x)

        return x

    # -------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print('Starting main routine...')

    #
    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # Which distances and sample size are we using?
    distances = '0100_0300'
    sample_size = '4k'

    # Where does our data live and which file should we use?
    data_path = '../data/'
    file_name = 'training_{}_{}.h5'.format(distances, sample_size)

    file_path = os.path.join(data_path, 'training', file_name)

    #
    # -------------------------------------------------------------------------
    # LOAD DATA, SPLIT TRAINING AND TEST SAMPLE, AND CREATE DATALOADERS
    # -------------------------------------------------------------------------

    print('Reading in data...', end=' ')

    # Load the data from the HDF file, split it, and convert to TensorDatasets
    tensor_datasets = load_data_as_tensor_datasets(file_path)
    data_train, data_test, data_validation = tensor_datasets

    print('Done!')

    #
    # -------------------------------------------------------------------------
    # SET UP A LOGGER FOR TENSORBOARD VISUALIZATION
    # -------------------------------------------------------------------------

    now = datetime.datetime.now()
    writer = SummaryWriter(log_dir='logs/{:%Y-%m-%d_%H:%M:%S}'.format(now))

    #
    # -------------------------------------------------------------------------
    # SET UP THE NET
    # -------------------------------------------------------------------------

    # Set up the net and make it CUDA ready; activate GPU parallelization
    net = Net()
    net.float().cuda()
    net = torch.nn.DataParallel(net)

    # If desired, load weights from pretrained model
    # net.load_state_dict(torch.load('pytorch_model_weights_100_300_4k.net'))

    # Set up the optimizer and the initial learning rate, and zero parameters
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    optimizer.zero_grad()

    # Set the mini-batch size, and calculate the number of mini-batches
    batch_size = 16
    n_minibatches_train = np.ceil(len(data_train) / batch_size)
    n_minibatches_test = np.ceil(len(data_test) / batch_size)
    n_minibatches_validation = np.ceil(len(data_validation) / batch_size)

    # Fix the number of epochs to train for
    n_epochs = 30

    #
    # -------------------------------------------------------------------------
    # TRAIN THE NET FOR THE GIVEN NUMBER OF EPOCHS
    # -------------------------------------------------------------------------

    print('\nStart training: Training on {} examples, validating on {} '
          'examples\n'.format(len(data_train), len(data_validation)))

    # -------------------------------------------------------------------------

    for epoch in range(n_epochs):

        print('Epoch {}/{}'.format(epoch+1, n_epochs))

        running_loss = 0
        running_hamm = 0
        start_time = time.time()

        #
        # ---------------------------------------------------------------------
        # LOOP OVER MINI-BATCHES AND TRAIN THE NETWORK
        # ---------------------------------------------------------------------

        for mb_idx, mb_data in enumerate(DataLoader(data_train,
                                                    batch_size=batch_size)):

            # Get the inputs and wrap them in a PyTorch variable
            inputs, labels = mb_data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # Get the size of the mini-batch
            mb_size = len(labels)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass through the net and reshape outputs properly
            outputs = net.forward(inputs)
            outputs = outputs.view((outputs.size()[0], outputs.size()[-1]))

            # Calculate weights and set up the loss function
            weights = torch.from_numpy(np.array([create_weights(_) for _ in
                                                 labels.data.cpu().numpy()]))
            loss_function = nn.BCELoss(weight=weights.float().cuda(),
                                       size_average=True).cuda()

            # Calculate the loss
            loss = loss_function(outputs, labels)
            running_loss += float(loss.data.cpu().numpy())

            # Use back-propagation to update the weights according to the loss
            loss.backward()
            optimizer.step()

            # Calculate the hamming distance between prediction and truth
            running_hamm += hamming_dist(outputs.data.cpu().numpy(),
                                         labels.data.cpu().numpy())

            # Make output to the command line
            progress_bar(current_value=mb_idx+1,
                         max_value=n_minibatches_train,
                         start_time=start_time,
                         loss=running_loss/(mb_idx+1),
                         hamming_dist=running_hamm/(mb_idx+1))

        #
        # ---------------------------------------------------------------------
        # LOOP OVER MINI-BATCHES AND EVALUATE ON VALIDATION SAMPLE
        # ---------------------------------------------------------------------

        # At the end of an epoch, calculate the validation loss
        val_loss = 0
        val_hamm = 0

        # Process validation data in mini-batches
        for mb_idx, mb_data in enumerate(DataLoader(data_validation,
                                                    batch_size=batch_size)):

            # Calculate the loss for a particular mini-batch
            inputs, labels = mb_data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # Forward pass through the net and reshape outputs properly
            outputs = net.forward(inputs)
            outputs = outputs.view((outputs.size()[0], outputs.size()[-1]))

            # Get the size of the mini-batch
            mb_size = len(labels)

            # Calculate weights and set up the loss function
            weights = torch.from_numpy(np.array([create_weights(_) for _ in
                                                 labels.data.cpu().numpy()]))
            loss_function = nn.BCELoss(weight=weights.float().cuda(),
                                       size_average=True).cuda()

            # Calculate the loss
            loss = loss_function(outputs, labels)
            val_loss += float(loss.data.cpu().numpy())

            # Calculate the hamming distance between prediction and truth
            val_hamm += hamming_dist(outputs.data.cpu().numpy(),
                                     labels.data.cpu().numpy())

        #
        # ---------------------------------------------------------------------
        # PRINT FINAL PROGRESS AND LOG STUFF FOR TENSORBOARD VISUALIZATION
        # ---------------------------------------------------------------------

        # Plot the final progress bar for this epoch
        progress_bar(current_value=n_minibatches_train,
                     max_value=n_minibatches_train,
                     start_time=start_time,
                     loss=running_loss/n_minibatches_train,
                     hamming_dist=running_hamm/n_minibatches_train,
                     val_loss=val_loss/n_minibatches_validation,
                     val_hamming_dist=val_hamm/n_minibatches_validation)
        print()

        # Save everything to the TensorBoard logger
        def log(name, value):
            writer.add_scalar(name, value, epoch)

        log('loss', running_loss/n_minibatches_train)
        log('hamming_dist', running_hamm/n_minibatches_train)
        log('val_loss', val_loss/n_minibatches_validation)
        log('val_hamming_dist', val_hamm/n_minibatches_validation)

    # -------------------------------------------------------------------------

    print('Finished Training!')
    writer.close()

    # Save the trained model
    print('Saving model...', end=' ')
    weights_file = 'pytorch_model_weights_{}_{}.net'.format(distances,
                                                            sample_size)
    torch.save(net.state_dict(), weights_file)
    print('Done!')

    #
    # -------------------------------------------------------------------------
    # MAKE PREDICTIONS ON THE TEST SET
    # -------------------------------------------------------------------------

    print('Start making predictions on the test sample...', end=' ')

    # Convert test data to numpy arrays that can be stored in an HDF file
    x_test = data_test.data_tensor.cpu().numpy()
    y_test = data_test.target_tensor.cpu().numpy()

    # Initialize an empty array for our predictions
    y_pred = np.empty((0, data_test.target_tensor.size()[1]))

    # Loop over the test set (in mini-batches) to get the predictions
    for mb_idx, mb_data in enumerate(DataLoader(data_test,
                                                batch_size=batch_size)):

        # Calculate the loss for a particular mini-batch
        inputs, labels = mb_data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # Make predictions for the given mini-batch
        outputs = net.forward(inputs)
        outputs = outputs.view((outputs.size()[0], outputs.size()[-1]))
        outputs = outputs.data.cpu().numpy()

        # Stack that onto the previous predictions
        y_pred = np.concatenate((y_pred, outputs), axis=0)

    # Set up the name and directory of the file where the predictions will
    # be saved.
    test_predictions_file = 'test_predictions_{}_{}.h5'.format(distances,
                                                               sample_size)
    test_predictions_path = os.path.join(data_path, 'predictions',
                                         test_predictions_file)

    with h5py.File(test_predictions_path, 'w') as file:

        file['x'] = x_test
        file['y_pred'] = y_pred
        file['y_true'] = y_test

    print('Done!')
