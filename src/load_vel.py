import numpy as np
import torch
import h5py
import os

def overthrust_model(input_data):

    strName    = 'overthrust_model.h5'

    m = np.transpose(h5py.File(os.path.join(input_data, 'overthrust_model.h5'), 'r')['m'][...])
    m0 = np.transpose(h5py.File(os.path.join(input_data, 'overthrust_model.h5'), 'r')['m0'][...])
    spacing = h5py.File(os.path.join(input_data, 'overthrust_model.h5'), 'r')['d'][...]
    shape = h5py.File(os.path.join(input_data, 'overthrust_model.h5'), 'r')['n'][...]

    origin = (0., 0.)
    dm = m - m0
    dm = torch.from_numpy(dm).unsqueeze_(0).unsqueeze_(0)
    
    return m0, m, dm, spacing, shape, origin