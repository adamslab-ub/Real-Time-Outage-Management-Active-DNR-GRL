import numpy as np
import gym
from stable_baselines3 import PPO
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import pickle
from stable_baselines3.common.utils import set_random_seed
import math
from Configs.training_config import get_training_config
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
#torch.manual_seed(1234)

def learning_rate_schedule(initial_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:
        decay_rate = 3
        return initial_value * math.exp((-(1-progress_remaining)**2*decay_rate))

        # return  initial_value
    return func

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = DSS_OutCtrl_Env()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    env_size = 123

    if env_size == 13:
        from Environments.DSSdirect_13bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
        from Policies.bus_34.Feature_Extractor import CustomGNN
        from Policies.bus_13.CustomPolicies import ActorCriticGCAPSPolicy
    elif env_size == 34: # will add more conditions once the 13 bus is fixed
        from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
        from Policies.bus_34.Feature_Extractor import CustomGNN
        from Policies.bus_34.CustomPolicies import ActorCriticGCAPSPolicy
    elif env_size == 123: # will add more conditions once the 13 bus is fixed
        from Environments.DSSdirect_123bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
        from Policies.bus_123.Feature_Extractor import CustomGNN
        from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy

    training_config = get_training_config()
    num_cpu = training_config.num_cpu
    use_cuda = training_config.use_cuda #torch.cuda.is_available() 
    training_config.device = torch.device("cuda:0" if training_config.use_cuda else "cpu")
    tb_logger_location = training_config.logger + training_config.node_encoder
    save_prefix = training_config.model_save+training_config.node_encoder+"_"+str(env_size)+"_"

    checkpoint_callback = CheckpointCallback(save_freq=training_config.save_freq, save_path=training_config.model_save,
                                         name_prefix=training_config.node_encoder)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    #env = DSS_OutCtrl_Env()

    if training_config.node_encoder == "CAPAM":

        policy_kwargs = dict(
            features_extractor_class=CustomGNN,
            features_extractor_kwargs=dict(features_dim=training_config.features_dim,node_dim=3),
            #activation_fn=torch.nn.Sigmoid,
            net_arch=[
                dict(
                    pi=
                    [training_config.features_dim,
                     2*training_config.features_dim,
                     2*training_config.features_dim,
                     training_config.features_dim],
                    vi=[training_config.features_dim,
                     2*training_config.features_dim,
                     2*training_config.features_dim,
                     training_config.features_dim]
                )],
            device=training_config.device
            # optimizer_class = th.optim.RMSprop,
            # optimizer_kwargs = dict(alpha=0.89, eps=rms_prop_eps, weight_decay=0)
        )

    else:

        policy_kwargs = dict(
            # features_extractor_class=CustomGNN,
            # features_extractor_kwargs=dict(features_dim=training_config.features_dim, node_dim=3),
            # activation_fn=torch.nn.Sigmoid,
            net_arch=[
                dict(
                    pi=
                    [training_config.features_dim,
                     2 * training_config.features_dim,
                     2 * training_config.features_dim,
                     training_config.features_dim],
                    vi=[training_config.features_dim,
                        2 * training_config.features_dim,
                        2 * training_config.features_dim,
                        training_config.features_dim]
                )]
            # device=training_config.device
            # optimizer_class = th.optim.RMSprop,
            # optimizer_kwargs = dict(alpha=0.89, eps=rms_prop_eps, weight_decay=0)
        )


    model = PPO(
        policy=ActorCriticGCAPSPolicy if training_config.node_encoder == "CAPAM" else "MultiInputPolicy",
        env=env,
        tensorboard_log=tb_logger_location, 
        policy_kwargs=policy_kwargs, 
        verbose=1,
        n_steps=training_config.n_steps,
        batch_size=training_config.batch_size,
        gamma=training_config.gamma,
        learning_rate=learning_rate_schedule(training_config.learning_rate),
        ent_coef=training_config.ent_coef
        ).learn(total_timesteps=training_config.total_steps, callback=checkpoint_callback)


    model.save(save_prefix+"_final")


# code to compute the number of trainable parameters
# pp=0
# for p in list(self.policy.parameters()):
#     nn=1
#     for s in list(p.size()):
#         nn = nn*s
#     pp += nn
# print(pp)
