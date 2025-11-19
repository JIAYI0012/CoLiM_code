# This file is adapted from:
# https://github.com/txie-93/cgcnn.git

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


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



# class GraphAttentionLayer(nn.Module):
#     """
#     Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat

#         self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)

#         self.leakyrelu = nn.LeakyReLU(self.alpha)

#     def forward(self, h, adj):
#         Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
#         e = self._prepare_attentional_mechanism_input(Wh)

#         zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, Wh)

#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime

#     def _prepare_attentional_mechanism_input(self, Wh):
#         # Wh.shape (N, out_feature)
#         # self.a.shape (2 * out_feature, 1)
#         # Wh1&2.shape (N, 1)
#         # e.shape (N, N)
#         Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
#         Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
#         # broadcast add
#         e = Wh1 + Wh2.T
#         return self.leakyrelu(e)

#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# class GINConv(MessagePassing):
#     def __init__(self, emb_dim, edge_attr_len):

#         super(GINConv, self).__init__(aggr="add")

#         self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
#                                  nn.Linear(emb_dim, emb_dim))
#         self.eps = nn.Parameter(torch.Tensor([0]))
#         self.bond_encoder = nn.Linear(edge_attr_len, emb_dim)

#     def forward(self, x, edge_index, edge_attr):
#         edge_embedding = self.bond_encoder(edge_attr)
#         out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
#         return out

#     def message(self, x_j, edge_attr):
#         return torch.sigmoid(x_j) * x_j + edge_attr

#     def update(self, aggr_out):
#         return aggr_out

