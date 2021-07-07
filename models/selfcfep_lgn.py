# -*- coding: utf-8 -*-
# @Time   : 2021/05/17
# @Author : Zhou xin
# @Email  : enoche.chow@gmail.com

r"""
################################################
Self-supervised CF

Using the same implementation of LightGCN in BUIR
With regularization


SELFCF_{ep}: edge pruning
"""

import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lightgcn import LightGCN
from models.common.abstract_recommender import GeneralRecommender

class SELFCFEP_LGN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SELFCFEP_LGN, self).__init__(config, dataset)
        self.user_count = self.n_users
        self.item_count = self.n_items
        self.latent_size = config['embedding_size']
        self.dropout = config['dropout']

        self.online_encoder = LightGCN(config, dataset)
        self.predictor = nn.Linear(self.latent_size, self.latent_size)
        self.norm_adj_matrix = self.online_encoder.norm_adj_matrix.clone().to(self.device)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col+self.n_users),
                             [1]*inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row+self.n_users, inter_M_t.col),
                                  [1]*inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def sparse_dropout(self, x):
        kprob = 1 - self.dropout
        randx = torch.rand(x._values().size()).to(self.device)
        mask = ((randx + kprob).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape).to(self.device)


    def forward(self, inputs):
        users, items = inputs[0], inputs[1]
        u_online, i_online = self.online_encoder()
        with torch.no_grad():
            u_target, i_target = u_online.clone(), i_online.clone()
            # edge pruning
            x = self.sparse_dropout(self.norm_adj_matrix)
            all_embeddings = torch.cat([u_target, i_target], 0)
            all_embeddings = torch.sparse.mm(x, all_embeddings)
            u_target = all_embeddings[:self.user_count, :]
            i_target = all_embeddings[self.user_count:, :]
            u_target.detach()
            i_target.detach()

        return self.predictor(u_online[users, :]), u_target[users, :], self.predictor(i_online[items, :]), i_target[items, :]

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def loss_fn(self, p, z):  # negative cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def calculate_loss(self, interaction):
        u_online, u_target, i_online, i_target = self.forward(interaction)

        loss_ui = self.loss_fn(u_online, i_target)/2
        loss_iu = self.loss_fn(i_online, u_target)/2

        return loss_ui + loss_iu

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, u_target, i_online, i_target = self.get_embedding()
        score_mat_ui = torch.matmul(u_online[user], i_target.transpose(0, 1))
        score_mat_iu = torch.matmul(u_target[user], i_online.transpose(0, 1))
        scores = score_mat_ui + score_mat_iu

        return scores

