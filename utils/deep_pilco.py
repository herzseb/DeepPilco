import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


class DeepPilco:
    def __init__(self, config):
        self.policy = PolicyModel(input_size=config["train"]["input_size_policy"], hidden_size=config["train"]["hidden_size_policy"], hidden_layer=config["train"]
                                  ["hidden_layer_policy"], output_size=config["train"]["output_size_policy"], dropout_rate=config["train"]["dropout_rate_policy"])
        self.dynamics_model = DynamicsModel(input_size=config["train"]["input_size_dynamic"], hidden_size=config["train"]["hidden_size_dynamic"],
                                            hidden_layer=config["train"]["hidden_layer_dynamic"], output_size=config["train"]["output_size_dynamic"])
        self.config = config


    # def init_policy_parameters(self):
    #     pass
    #     # 2 initialise policy parameters randomly

    def sample_particles_from_init(self):
        return np.random.normal(loc=self.config["env"]["init_mean_position"], scale=self.config["env"]["init_std_position"], size=np.zeros(self.config["train"]["K"]))

    def sample_particles(self, mean, std):
            return np.random.normal(loc=mean, scale=std, size=np.zeros(self.config["train"]["K"]))


    def sample_masks(self):
        return np.random.Generator.integers(low=1000, high=9999, size=self.config["train"]["K"])

    def predict_trajectories(self, config):
        # Initialise set of K particles
        self.particles = self.sample_particles_from_init()
        # Sample BNN dynamics model weights Wk for each k (i.e. sample seed values for dropout mask)
        self.masks = self.sample_masks()
        # save trajectory
        trajectory = []
        # for each time step of T
        for i in range(self.config["train"]["T"]):
            posteriors = []
            # for each particle
            for particle, mask in zip(self.particles, self.masks):
                # get action from policy
                action = self.policy(particle)
                # get posterior of weights W
                delta_x = self.dynamics_model(x=particle, action=action, dropout_rate=self.config["train"]["dropout_sampling_dynamic"], seed=mask)
                y = particle + delta_x
                posteriors.append(y)
            # fit one gaussian with mean and standard deviation from posterior
            mean = np.mean(posteriors)
            std = np.std(posteriors)
            # sample set of particles from gaussian
            self.particles = self.sample_particles(mean=mean, std=std)
            trajectory.append(trajectory)
        return trajectory

                 

    def sample_trajectory(self, env, T):
        rollout = []
        state = env.observation()
        for i in range(T):
            action = self.get_action(state)
            next_state = env.step(action)
            diff = next_state - state
            rollout.append(
                {
                    "state": state,
                    "action": action,
                    "diff": diff,
                }
            )
            state = next_state
        return rollout

    def cost(self, state):
        #FIXME arg of upmost point
        d = np.sqrt((np.sin(state[1])*self.config["env"]["l1"] - self.config["env"]["l1"])**2 + (np.cos(state[1])*self.config["env"]["l1"] - 0)**2)
        return 1 - np.exp(-0.5*(d**2)/self.config["env"]["cost_sigma"]**2)

    def evaluate_policy(self, trajectory):
        #FIXME expected cost
        J = 0
        for state in trajectory[::-1]:
            J = self.config["train"]["discount"]*J + self.cost(state)
        return J

    def requires_grad(self, model, require_grad):
        for param in model.parameters():
            param.requires_grad = require_grad


class DynamicsModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(DynamicsModel, self).__init__()
        self.mlp_block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.mlp = nn.ModuleList()
        self.mlp.append([self.mlp_block for i in range(hidden_layer)])
        self.final_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, action, dropout_rate, seed=None):
        if seed==None:
            seed = np.random.Generator.integers(low=1000, high=9999, size=1)
        x = torch.stack((x,action), dim=1)
        dropout = nn.Dropout(p=dropout_rate, inplace=True)
        for layer in range(len(self.mlp)):
            x = self.mlp[layer](x)
            torch.manual_seed(seed)
            x = dropout(x)
        x = self.final_layer(x)
        return x

    






class PolicyModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size, dropout_rate):
        super(PolicyModel, self).__init__()
        self.mlp_block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.mlp = nn.ModuleList()
        self.mlp.append([self.mlp_block for i in range(hidden_layer)])
        self.final_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.mlp(x)
        x = self.final_layer(x)
        return x
    
    def update(self, optimizer, cost):
        cost.backwards()
        optimizer.step()


class RolloutDataset(Dataset):
    def __init__(self, trajectory):
        self.trajectory = trajectory

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, idx):
        item = self.trajectory[idx]
        return item["state"], item["action"], item["diff"]

        