import numpy as np
import gym
from stable_baselines3 import PPO
import torch
from stable_baselines3 import A2C, PPO
from Policies.bus_13.CustomPolicies import ActorCriticGCAPSPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import pickle
from stable_baselines3.common.utils import set_random_seed

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

class CustomGNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self,
                observation_space: gym.spaces.Box,
                 features_dim: int = 256,
                 n_layers=2,
                 n_dim=256,
                 n_p=2,
                 node_dim=3,
                 n_K=1,
                 ):
        super(CustomGNN, self).__init__(observation_space, features_dim)
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = torch.nn.Linear(node_dim, n_dim * n_p)
        self.W_L_1_G1 = torch.nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_L_1_G2 = torch.nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)
        self.W_F = torch.nn.Linear(n_dim * n_p, features_dim)
        self.full_context_nn = torch.nn.Sequential(*[torch.nn.Linear(19, features_dim), torch.nn.Linear(features_dim, features_dim)])
        self.switch_encoder = torch.nn.Sequential(*[torch.nn.Linear(16, features_dim), torch.nn.Linear(features_dim, features_dim)])

        self.activ = torch.nn.LeakyReLU()

    def forward(self, data):
        X = data['NodeFeat(BusVoltage)']
        num_samples, num_locations, _ = X.size()
        A = data["Adjacency"]
        # print(A.shape)
        D = torch.mul(torch.eye(num_locations).expand((num_samples, num_locations, num_locations)).to(A.device),
                      (A.sum(-1))[:, None].expand((num_samples, num_locations, num_locations)))

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)
        # print(torch.isnan(X).to(torch.int32).sum())
        #print(X)
        # K = 3
        L = D - A
        
        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :]
                                          ),
                                         -1))
        g_L1_2 = self.W_L_1_G2(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :]
                                          ),
                                         -1))

        F1 = torch.cat((g_L1_1, g_L1_2), -1)
        # F1 = self.activ(F1)

        F_final = self.W_F(F1)

        h = F_final  # torch.cat((init_depot_embed, F_final), 1)
        # return (
        #     h,  # (batch_size, graph_size, embed_dim)
        #     h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        # )
        switch_embeddings = self.switch_encoder(h.permute(0,2,1))
       
        context = self.full_context_nn(torch.cat((data["EnergySupp"],data["VoltageViolation"], data["EdgeFeat(Branchflow)"]), -1))
        
        final = switch_embeddings.mean(dim=1)+context
        
        return final

