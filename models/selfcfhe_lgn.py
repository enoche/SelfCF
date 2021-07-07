# -*- coding: utf-8 -*-
# @Time   : 2021/05/17
# @Author : Zhou xin
# @Email  : enoche.chow@gmail.com

r"""
################################################
Self-supervised CF

Using the same implementation of LightGCN in BUIR
With regularization


SELFCF_{he}: history embeddings
"""

import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lightgcn import LightGCN
from models.common.abstract_recommender import GeneralRecommender

class SELFCFHE_LGN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SELFCFHE_LGN, self).__init__(config, dataset)
        self.user_count = self.n_users
        self.item_count = self.n_items
        self.latent_size = config['embedding_size']
        self.momentum = config['momentum']

        self.online_encoder = LightGCN(config, dataset)
        self.predictor = nn.Linear(self.latent_size, self.latent_size)
        self.u_target_his = torch.randn((self.n_users, self.latent_size), requires_grad=False).to(self.device)
        self.i_target_his = torch.randn((self.n_items, self.latent_size), requires_grad=False).to(self.device)

    def forward(self, inputs):
        users, items = inputs[0], inputs[1]
        u_online, i_online = self.online_encoder()
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        with torch.no_grad():
            u_target, i_target = self.u_target_his.clone()[users, :], self.i_target_his.clone()[items, :]
            u_target.detach()
            i_target.detach()

            u_target = u_target * self.momentum + u_online.data * (1. - self.momentum)
            i_target = i_target * self.momentum + i_online.data * (1. - self.momentum)

            self.u_target_his[users, :] = u_online.clone()
            self.i_target_his[items, :] = i_online.clone()

        return self.predictor(u_online), u_target, self.predictor(i_online), i_target

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

