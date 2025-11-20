from __future__ import print_function, division
import functools
import os
import random
import csv
import json
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure, Molecule
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from utilities import group_train_val_test_split

def get_data_loader(dataset,
                    collate_fn=default_collate,
                    batch_size=64,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=False):
   
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn,
                             pin_memory=pin_memory)
    return data_loader


def get_grouped_train_val_test_loaders(
    dataset,
    group_ids,             # len(dataset)
    batch_size=64,
    val_ratio=0.1,
    test_ratio=0.1,
    num_workers=1,
    pin_memory=False,
    collate_fn=None,
    seed=42,
):

    assert len(dataset) == len(group_ids)

    train_idx, val_idx, test_idx = group_train_val_test_split(
        group_ids=group_ids,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader




def collate_pool_CLNN(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea_c, nbr_fea_c, nbr_fea_idx_c,
        ligand_fea)

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea_c: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea_c: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx_c: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    core_atom_idx: list of torch.LongTensor of length N0

    batch_ligand_fea: torch.Tensor shape (N, orig_atom_fea_len(512))
      Atom features from atom type
        
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_xyz_ids: list
    """
    batch_atom_fea_c, batch_nbr_fea_c, batch_nbr_fea_idx_c = [], [], []
    batch_ligand_fea = []
    core_atom_idx, batch_target = [], []
    batch_ids = []
    base_idx_c = 0
    
    for i, ((atom_fea_c, nbr_fea_c, nbr_fea_idx_c), ligand_fea, target, (core_id, ligand_id))\
            in enumerate(dataset_list):
        # collate for cores
        n_i_c = atom_fea_c.shape[0]  # number of atoms for this core
        batch_atom_fea_c.append(atom_fea_c)
        batch_nbr_fea_c.append(nbr_fea_c)
        batch_nbr_fea_idx_c.append(nbr_fea_idx_c+base_idx_c)
        new_idx_c = torch.LongTensor(np.arange(n_i_c)+base_idx_c)
        core_atom_idx.append(new_idx_c)
        base_idx_c += n_i_c

        # collate for ligands
        batch_ligand_fea.append(ligand_fea)
        
        
        batch_target.append(target)
        batch_ids.append([core_id, ligand_id])
        
    return (torch.cat(batch_atom_fea_c, dim=0),
            torch.cat(batch_nbr_fea_c, dim=0),
            torch.cat(batch_nbr_fea_idx_c, dim=0),
            core_atom_idx,
            torch.stack(batch_ligand_fea, dim=0)),\
        torch.stack(batch_target, dim=0),\
        batch_ids

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)






class CLNN_dataset(Dataset):
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        
        
        core_id, ligand_id, target = self.id_prop_data[idx]
        core = np.load(os.path.join(self.root_dir+'/Core_npz_cv2',
                                        core_id+'.npz'))
        ligand_fea = np.load(os.path.join(self.root_dir+'/mol_cv2',
                                          ligand_id+'.npy'))
        atom_fea_c = core['atom_fea']
        nbr_fea_c = core['nbr_fea']
        nbr_fea_idx_c = core['nbr_fea_idx']

        atom_fea_c = torch.Tensor(atom_fea_c)
        nbr_fea_c = torch.Tensor(nbr_fea_c)
        nbr_fea_idx_c = torch.LongTensor(nbr_fea_idx_c)

        ligand_fea = torch.Tensor(ligand_fea)
        target = int(target)
        target = torch.tensor(target, dtype=torch.long)
        return (atom_fea_c, nbr_fea_c, nbr_fea_idx_c), ligand_fea, target, (core_id, ligand_id)
    

