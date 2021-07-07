# -*- coding: utf-8 -*-
# @Time   : 2021/05/17
# @Author : Zhou xin
# @Email  : enoche.chow@gmail.com

r"""
BUIR_NB
################################################
Bootstrapping User and Item Representations for One-Class Collaborative Filtering, SIGIR21
"""

import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.abstract_recommender import GeneralRecommender

class BUIR_NB(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BUIR_NB, self).__init__(config, dataset)
        self.user_count = self.n_users
        self.item_count = self.n_items
        self.latent_size = config['embedding_size']
        self.momentum = config['momentum']

        self.online_encoder = LGCN_Encoder(config, dataset)
        self.target_encoder = LGCN_Encoder(config, dataset)

        self.predictor = nn.Linear(self.latent_size, self.latent_size)

        self._init_target()

    def _init_target(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    def _update_target(self):
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def forward(self, inputs):
        self._update_target()

        u_online, i_online = self.online_encoder(inputs)
        u_target, i_target = self.target_encoder(inputs)
        return self.predictor(u_online), u_target, self.predictor(i_online), i_target

    @torch.no_grad()
    def get_embedding(self):
        u_online, i_online = self.online_encoder.get_embedding()
        return self.predictor(u_online), u_online, self.predictor(i_online), i_online

    def calculate_loss(self, interaction):
        u_online, u_target, i_online, i_target = self.forward(interaction)

        u_online = F.normalize(u_online, dim=-1)
        u_target = F.normalize(u_target, dim=-1)
        i_online = F.normalize(i_online, dim=-1)
        i_target = F.normalize(i_target, dim=-1)

        loss_ui = 2 - 2 * (u_online * i_target).sum(dim=-1)
        loss_iu = 2 - 2 * (i_online * u_target).sum(dim=-1)

        return (loss_ui + loss_iu).mean()

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, u_target, i_online, i_target = self.get_embedding()
        score_mat_ui = torch.matmul(u_online[user], i_target.transpose(0, 1))
        score_mat_iu = torch.matmul(u_target[user], i_online.transpose(0, 1))
        scores = score_mat_ui + score_mat_iu

        return scores


class LGCN_Encoder(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LGCN_Encoder, self).__init__(config, dataset)
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        self.user_count = self.n_users
        self.item_count = self.n_items
        self.latent_size = config['embedding_size']
        self.layers = [self.latent_size] * config['n_layers']

        self.drop_ratio = 1.0
        self.drop_flag = True

        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = self.get_norm_adj_mat().to(self.device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.user_count, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.item_count, self.latent_size)))
        })

        return embedding_dict

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

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(self.device)
        return out * (1. / (1 - rate))

    def forward(self, inputs):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    np.random.random() * self.drop_ratio,
                                    self.sparse_norm_adj._nnz()) if self.drop_flag else self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        users, items = inputs[0], inputs[1]
        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings

    @torch.no_grad()
    def get_embedding(self):
        A_hat = self.sparse_norm_adj

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        return user_all_embeddings, item_all_embeddings


