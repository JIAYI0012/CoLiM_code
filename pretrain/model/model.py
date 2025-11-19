# This file is adapted from:
# https://github.com/txie-93/cgcnn.git



from __future__ import print_function, division

import torch
import torch.nn as nn
import torch
import torch.nn.functional as F


from .layers import ConvLayer

import torch

import os
import os.path as osp
from functools import partial
from math import pi as PI
from math import sqrt
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Embedding, Linear

class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding_core = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs_core = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc_core = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            # self.fc_out = nn.Linear(h_fea_len, 2)
            self.fc_out = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len//2),
                                        nn.Softplus(),
                                        nn.Linear(h_fea_len//2, 2)])
        else:
            # self.fc_out = nn.Linear(h_fea_len, 1)
            self.fc_out = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len//2),
                                        nn.Softplus(),
                                        nn.Linear(h_fea_len//2, 1)])
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding_core(atom_fea)
        identity = atom_fea
        for conv_func in self.convs_core:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
            atom_fea = atom_fea + identity
            identity = atom_fea
        crys_fea = self.pooling(atom_fea, crystal_atom_idx) #tensor with shape(N0, fea_len)
        crys_fea = self.conv_to_fc_core(self.conv_to_fc_softplus(crys_fea)) #(N0, h_len)
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = crys_fea
        repr = crys_fea
        for layer in self.fc_out:
            out = layer(out)
        if self.classification:
            out = self.logsoftmax(out)
        return out, repr

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0] #whether the number of atoms in atom_fea is agree with len of one tensor in crystal_atom_idx
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx] #summed_fea list of No with shape(fea_len), keep dim
        return torch.cat(summed_fea, dim=0) #tensor with shape(NO, fea_len)

