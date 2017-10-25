# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import os
import sys
import h5py
import torch
import torch.nn as nn

from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

sys.path.insert(0, '../train/')
from models import TimeSeriesFCN


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_data_as_tensor_datasets(file_path, random_seed=42):

    # Set the seed for the random number generator
    np.random.seed(random_seed)

    # Read in the spectrograms from the HDF file
    with h5py.File(file_path, 'r') as file:
        x = np.array(file['x'])[:10]
        y = np.array(file['y_true'])[:10]

    # Convert to torch Tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    # Create TensorDatasets for training, test and validation
    tensor_dataset = TensorDataset(x, y)

    return tensor_dataset


# -----------------------------------------------------------------------------


def apply_model(model, data_loader, as_numpy=False):

    # Initialize an empty array for our predictions
    y_pred = []

    # Loop over the test set (in mini-batches) to get the predictions
    for mb_idx, mb_data in enumerate(data_loader):

        print(mb_idx)

        # Get the inputs and wrap them in a PyTorch variable
        inputs, labels = mb_data
        inputs = Variable(inputs, volatile=True)
        labels = Variable(labels, volatile=True)

        # If CUDA is available, run everything on the GPU
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        # Make predictions for the given mini-batch
        outputs = model.forward(inputs)
        outputs = outputs.view((outputs.size()[0], outputs.size()[-1]))

        # Stack that onto the previous predictions
        y_pred.append(outputs)

    # Concatenate the list of Variables to one Variable (this is faster than
    # concatenating all intermediate results) and make sure results are float
    y_pred = torch.cat(y_pred, dim=0).float()

    # If necessary, convert model outputs to numpy array
    if as_numpy:
        y_pred = y_pred.data.cpu().numpy()

    return y_pred


# -----------------------------------------------------------------------------


def get_weights(labels, threshold):
    weights = torch.eq(torch.gt(labels, 0) * torch.lt(labels, threshold), 0)
    return weights.float()


# -----------------------------------------------------------------------------


def loss_function(y_pred, y_true, weights):

    # Set up the Binary Cross-Entropy term of the loss
    bce_loss = nn.BCELoss(weight=weights)
    if torch.cuda.is_available():
        bce_loss = bce_loss.cuda()

    return bce_loss(y_pred, y_true)


# -----------------------------------------------------------------------------


def accuracy(y_true, y_pred):

    # Make sure y_pred is rounded to 0/1
    y_pred = torch.round(y_pred)

    result = torch.mean(torch.abs(y_true - y_pred), dim=1)
    result = torch.mean(result, dim=0)

    return 1 - float(result.data.cpu().numpy())


# -----------------------------------------------------------------------------
# MAIN PROGRAM
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # LOAD THE PRE-TRAINED MODEL
    # -------------------------------------------------------------------------

    # Initialize the model
    model = TimeSeriesFCN()

    # Define the weights we want to use (in this case form TCL)
    weights_file = os.path.join('..', 'train', 'weights',
                                'timeseries_weights_{}_{}_{}.net'.
                                format('GW170104', '0100_1200', '16k'))

    # Check if CUDA is available. If not, loading the weights is a bit more
    # cumbersome and we have to use some tricks
    if torch.cuda.is_available():
        model.float().cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(weights_file))
    else:
        state_dict = torch.load(weights_file,
                                map_location=lambda storage, loc: storage)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    # -------------------------------------------------------------------------
    # LOOP OVER THE DIFFERENT BASELINE SETS
    # -------------------------------------------------------------------------

    for dist in ['0100_0300', '0250_0500', '0400_0800', '0700_1200']:

        print('NOW EVALUATING BASELINE FOR:', dist)

        # Load data into data tensor and data loader
        file_path = os.path.join('..', 'data', 'predictions', 'timeseries',
                                 'baseline', 'predctions_{}_{}_{}.h5'.
                                 format('GW170104', dist, '8k'))
        datatensor = load_data_as_tensor_datasets(file_path)
        dataloader = DataLoader(datatensor, batch_size=32)

        # Get the true labels we need for the comparison
        true_labels = Variable(datatensor.target_tensor, volatile=True)

        # Get the predictions by applying the pre-trained net
        predictions = apply_model(model, dataloader)

        # Get weights that we need to calculate our metrics
        weights = get_weights(true_labels, 1.4141823e-22)

        # Calculate the loss (averaged over the entire dataset)
        loss = loss_function(y_pred=predictions,
                             y_true=torch.ceil(true_labels),
                             weights=weights)
        loss = float(loss.data.cpu().numpy())

        # Calculate the accuracy (averaged over the entire dataset)
        accuracy = accuracy(y_pred=predictions * weights,
                            y_true=torch.ceil(true_labels * weights))

        # Print the results
        print('Loss: {:.3f}'.format(loss))
        print('Accuracy: {:.3f}'.format(accuracy))
        print()
