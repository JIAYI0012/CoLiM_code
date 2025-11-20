from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import TransformerConv, SchNet, DimeNetPlusPlus, DimeNet




class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :] 
        #generate from (N, fea) and (N, M) to (N, M, fea), select the neigbor feature
        
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        #(N, M, fea_len) + (N, M, fea_len) + (N, M, nbr_fea) = (N, M, 2 * fea_len + nbr_fea_len) #atom + nbr_atom + edge

        total_gated_fea = self.fc_full(total_nbr_fea)
        #(N, M, 2 * fea_len + nbr_fea_len) to (N, M, 2 * fea_len)

        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        #Normalize the tensor by shape it to (N * M, 2 * fea_len) and reshape it back

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        #cut total_gated_fea into 2 tensor in dim2, (N, M, fea_len)

        nbr_filter = self.sigmoid(nbr_filter) #Eq5_firstpart
        nbr_core = self.softplus1(nbr_core) #Eq5_secondpart
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1) #EQ5_elementwise_product+SUM/Vi+的部分
        nbr_sumed = self.bn2(nbr_sumed) #linear
        out = self.softplus2(atom_in_fea + nbr_sumed) #vi+...
        return out
    





class CLNN(nn.Module):
  """
  core encoder: GCN 128
  ligand encoder: Unimol 512 + MLP: 512->128
  fine-tuning: train ligand linear layer and fc_out
  """
  def __init__(self, orig_atom_fea_len, nbr_fea_len, ligand_fea_len=512, droprate=0.5,
                atom_fea_len=64, n_conv=3, h_fea_len=128,
                classification=True):
    super(CLNN, self).__init__()
    self.classification = classification

    self.embedding_core = nn.Linear(orig_atom_fea_len, atom_fea_len)
    self.convs_core = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                nbr_fea_len=nbr_fea_len)
                                for _ in range(n_conv)])
    self.batchnorm_core = nn.BatchNorm1d(atom_fea_len)
    self.conv_to_fc_core = nn.Linear(atom_fea_len, h_fea_len)
    self.softplus = nn.Softplus()
    self.relu = nn.ReLU()

    self.linear_ligand = nn.ModuleList([nn.Linear(ligand_fea_len, ligand_fea_len//2),
                                        # nn.BatchNorm1d(ligand_fea_len//2),
                                        nn.Softplus(),
                                        nn.Linear(ligand_fea_len//2, h_fea_len)])
    
    
    self.batchnorm_cls = nn.BatchNorm1d(h_fea_len*2)

    if self.classification:
        self.fc_out = nn.ModuleList([nn.Linear(h_fea_len*2, h_fea_len),
                                    # nn.BatchNorm1d(h_fea_len),
                                    nn.Softplus(),
                                    nn.Linear(h_fea_len, 2)])
    else:
        self.fc_out = nn.ModuleList([nn.Linear(h_fea_len*2, h_fea_len),
                                    # nn.BatchNorm1d(h_fea_len),
                                    nn.Softplus(),
                                    nn.Linear(h_fea_len, 1)])
    if self.classification:
        self.dropout = nn.Dropout(p=droprate)

  def forward(self, atom_fea_c, nbr_fea_c, nbr_fea_idx_c, core_atom_idx,
               ligand_fea):
    
    atom_fea_c = self.embedding_core(atom_fea_c)
    c_identity = atom_fea_c
    for conv_func in self.convs_core:
      atom_fea_c = conv_func(atom_fea_c, nbr_fea_c, nbr_fea_idx_c)
      atom_fea_c = atom_fea_c + c_identity
      c_identity = atom_fea_c
    
    core_fea = self.pooling(atom_fea_c, core_atom_idx) 
    core_fea = self.softplus(self.conv_to_fc_core(core_fea))

    
    ligand_fea = ligand_fea.squeeze()
    for layer in self.linear_ligand:
       ligand_fea = layer(ligand_fea)
    
    structure_fea = self.batchnorm_cls(torch.cat((core_fea, ligand_fea), dim=-1)) #(N0, h_len * 2)
    out = self.dropout(structure_fea)
    for layer in self.fc_out:
      out = layer(out)
    
    return out

  def pooling(self, atom_fea, crystal_atom_idx):
    assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
        atom_fea.data.shape[0]
    summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                  for idx_map in crystal_atom_idx]
    return torch.cat(summed_fea, dim=0)
  
  def pooling_modified(self, atom_fea, crystal_atom_idx):
    """
    A modified pooling method
    """
    # Step1, gating weight sum pooling 
    weighted_sum_rep = []
    for idx in crystal_atom_idx:
        weighted_sum = torch.mean(atom_fea[idx], dim=0, keepdim=False)
        weighted_sum_rep.append(weighted_sum)
    weighted_sum_rep = torch.stack(weighted_sum_rep)
    
    # Step2, Max pooling
    max_pooling_rep = []
    for idx in crystal_atom_idx:
        max_pooling, _ = torch.max(atom_fea[idx], dim=0)
        max_pooling_rep.append(max_pooling)
    max_pooling_rep = torch.stack(max_pooling_rep)

    # Step3, concat the two rep
    graph_rep = torch.cat((weighted_sum_rep, max_pooling_rep), dim=1)

    return graph_rep