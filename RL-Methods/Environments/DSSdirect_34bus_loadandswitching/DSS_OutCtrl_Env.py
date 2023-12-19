import gym
import random
from gym import spaces
import numpy as np
import logging
from random import sample, choices, uniform, randint
from math import ceil
from Environments.DSSdirect_34bus_loadandswitching.DSS_Initialize import *
from Environments.DSSdirect_34bus_loadandswitching.state_action_reward import *
from gym.utils import seeding

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.WARNING)


class DSS_OutCtrl_Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print(
            "Initializing 34-bus env for outage management with generators, sectionalizing and tie switches (load control included)")
        self.DSSCktObj, G_init, conv_flag = initialize()  # the DSSCircuit is set up and initialized
        # Set up action and observation space variables        
        self.outedges = []  # to track the outage conditions(list of multi-line outage)
        self.G = G_init.copy()
        self.action_space = spaces.MultiBinary(n_actions)
        self.observation_space = spaces.Dict({"EnergySupp": spaces.Box(low=0, high=2, shape=(1,)),
                                              "NodeFeat(BusVoltage)": spaces.Box(low=0, high=2,
                                                                                 shape=(len(G_init.nodes()), 3)),
                                              "EdgeFeat(Branchflow)": spaces.Box(low=0, high=2,
                                                                                 shape=(len(G_init.edges()),)),
                                              "Adjacency": spaces.Box(low=0, high=1,
                                                                      shape=(len(G_init.nodes()), len(G_init.nodes()))),
                                              "VoltageViolation": spaces.Box(low=0, high=1000, shape=(1,)),
                                              "ConvergenceViolation": spaces.Box(low=0, high=1, shape=(1,)),
                                              "ActionMasking": spaces.Box(low=0, high=1, shape=(n_actions,))
                                              })
        print('Env initialized')

    def step(self, action):
        # Getting observation before action is executed
        observation = get_state(self.DSSCktObj, self.G, self.outedges)  # function to get state of the network
        # Executing the switching action
        try:
            self.DSSCktObj, self.G = take_action(action, self.outedges)  # function to implement the action
            # Getting observation after action is taken
            obs_post_action = get_state(self.DSSCktObj, self.G, self.outedges)
            reward = get_reward(obs_post_action)  # function to calculate reward
        except:
            obs_post_action = get_state(self.DSSCktObj, self.G, self.outedges)
            reward = np.array([0.0])
        done = True
        info = {"is_success": done,
                "episode": {
                    "r": reward,
                    "l": 1
                }
                }
        logging.info('Step success')
        return obs_post_action, reward[0], done, info

    def reset(self):
        # In reset function we simulate different line outage scenarios
        logging.info('resetting environment...')
        self.DSSCktObj, G_init, conv_flag = initialize()  # initial set up
        self.G = G_init.copy()

        # ---- Outage Scenario
        # Subgraph approach for outages
        max_rad = nx.diameter(G_init)  # maximum radius to create a subgraph
        max_percfail = 0.5  # Percentage of edges failed within subgraph -variable
        nd = random.choice(list(G_init.nodes()))  # select initial node around which subgraph is formed
        rad = ceil(uniform(0, max_rad / 2))  # select radius of subgraph from a uniform distribution

        # Failure isloation simulation do using base graph
        Gsub = nx.ego_graph(G_base, nd, radius=rad,
                            undirected=False)  # form subgraph around selected node with given radius
        sub_edges = list(Gsub.edges())  # list of subgraph edges
        out_perc = uniform(0, max_percfail)  # percentage of edge failure within subgraph
        N_out = math.ceil(len(sub_edges) * out_perc)  # number of edge outages within subgraph
        out_edges = sample(sub_edges,
                           k=N_out)  # random sampling without replacement # random failure of edges within subgraph

        for o_e in out_edges:
            (u,v)=o_e
            branch_name= G_init.edges[o_e]['label'][0]
            self.DSSCktObj.dss.Text.Command(f'Open {branch_name} term=1') #disable the outage line
            try:
                self.DSSCktObj.dss.Solution.Solve()
            except:
                return self.reset()

        self.G.remove_edges_from(out_edges)  # each instance of the graph includes the outage scenario
            
        self.outedges = out_edges  # outage scenario

        logging.info("reset complete\n")
        obs = get_state(self.DSSCktObj, self.G, self.outedges)
        if np.isnan(obs["EdgeFeat(Branchflow)"]).astype(np.int32).sum() > 0 or obs[
            "NodeFeat(BusVoltage)"].max() > 100000:  # if the scenario is causing nan as edge variables
            return self.reset()
        return obs

    def render(self, mode='human', close=False):
        pass
