import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = 'cpu'

# TODO reparameterization trick for moment matching
# correct gradient
# replay buffer?

class DeepPilco:
    def __init__(self, config, env, wandb):
        self.policy = PolicyModel(input_size=config["train"]["input_size_policy"], hidden_size=config["train"]["hidden_size_policy"], hidden_layer=config["train"]
                                  ["hidden_layer_policy"], output_size=config["train"]["output_size_policy"], dropout_rate=config["train"]["dropout_training_policy"], env=config["train"]["hidden_size_policy"])
        self.policy = self.policy.to(device=device)
        self.dynamics_model = DynamicsModel(input_size=config["train"]["input_size_dynamic"], hidden_size=config["train"]["hidden_size_dynamic"],
                                            hidden_layer=config["train"]["hidden_layer_dynamic"], output_size=config["train"]["output_size_dynamic"])
        self.dynamics_model = self.dynamics_model.to(device=device)
        self.config = config
        self.env = env
        self.wandb = wandb

    def sample_particles_from_init(self, env):
        # return np.random.normal(loc=self.config["env"]["init_mean_position"], scale=self.config["env"]["init_std_position"], size=self.config["train"]["K"])
        state = env.reset()
        if len(state.shape) > 1:
            state = state[0]
        state = torch.tensor(state, dtype=torch.float32)
        return state

    def sample_particles(self, posteriors):
        mean = torch.mean(posteriors, dim=0, keepdim=True)
        std = torch.std(posteriors, dim=0, keepdim=True)
        scale = torch.randn(self.config["train"]["K"], mean.size(1)).to(device)
        particles = mean + std * scale
        return particles

    def predict_trajectories(self):
        # init cost
        cost = 0
        # for batch in batchsize
        for j in range(self.config["train"]["policy_batch_size"]):
            # Initialise set of K particles
            self.particles = []
            
            for k in range(self.config["train"]["K"]):
                self.particles.append(
                    self.sample_particles_from_init(self.env).to(device=device))
            self.particles = torch.stack(self.particles)
            self.particles.requires_grad = True
            # Sample BNN dynamics model weights Wk for each k (i.e. sample seed values for dropout mask)
            self.masks = []
            with torch.no_grad():
                for k in range(self.config["train"]["K"]):
                    self.masks.append(CustomDropout(
                        dropout=self.config["train"]["dropout_sampling_dynamic"], size=self.config["train"]["hidden_size_dynamic"]))
            # save trajectory
            trajectory = torch.empty(
                (0, self.config["train"]["output_size_dynamic"]), dtype=torch.float32, device=device)
            trajectory.requires_grad=False
            # for each time step of T
            for i in range(self.config["train"]["T"]):
                posteriors = torch.empty(
                    (0, self.config["train"]["output_size_dynamic"]), dtype=torch.float32, device=device)
                # for each particle
                for particle, mask in zip(self.particles, self.masks):
                    # get action from policy
                    particle = torch.unsqueeze(particle, dim=0)
                    action = self.policy(particle, eps=0.1)
                    # get posterior of weights W
                    delta_x = self.dynamics_model(
                        x=particle, action=action, dropout=mask)
                    y = particle + delta_x
                    posteriors = torch.cat((posteriors, y), dim=0)
                # fit one gaussian with mean and standard deviation from posterior
                mean = torch.mean(posteriors, dim=0)
                std = torch.std(posteriors, dim=0)
                cost += self.cost(mean, std) * self.config["train"]["discount"]**i
                # sample set of particles from gaussian
                self.particles = self.sample_particles(posteriors)
                # self.particles = []
                # for k in range(self.config["train"]["K"]):
                #     self.particles.append(
                #         self.sample_particles(mean=mean, std=std))
                with torch.no_grad():
                    trajectory = torch.cat(
                            (trajectory, torch.unsqueeze(mean, dim=0)), dim=0)
        cost = cost / self.config["train"]["policy_batch_size"]
        return trajectory, cost

    def cost(self, state, std=None):
        l1 = self.config["env"]["l1"]
        l2 = self.config["env"]["l2"]
        if self.config["train"]["gym"] == 'InvertedDoublePendulum-v4':
            # d = torch.sqrt((state[3]*l1 + state[4]*l2 - (l1+l2))** 2 + 0.1 * (state[1]*l1 + state[2]*l2)**2)
            d = 2 - state[3] + state[4]
        elif self.config["train"]["gym"] == 'InvertedPendulum-v4':
            d = torch.sqrt((1 - torch.cos(state[1]))** 2 + 0.1 * (torch.sin(state[1]))**2)
        elif self.config["train"]["gym"] == 'SingleSwingUp':
            d = torch.sqrt((1 - torch.cos(state[2]))** 2 + 0.1 * (torch.sin(state[2]))**2)
        cost = 1 - torch.exp(-0.5*(d**2)/self.config["env"]["cost_sigma"]**2)
        # if std is not None:
        #     combined_var = std[1] + std[2] + std[3] + std[4]
        #     cost = cost * combined_var
        return cost

    def sample_trajectory(self, env, T):
        rollout = []
        state = env.reset()
        if len(state.shape) > 1:
            state = state[0]
        state = torch.tensor(state, dtype=torch.float32).to(device=device)
        for i in range(T):
            state = state.to(device=device)
            action = self.policy(state, eps=0.5)
            action = action.detach().to("cpu")
            self.wandb.log({"action": action})
            next_state = env.step(action)
            next_state = torch.tensor(next_state[0], dtype=torch.float32)
            state = state.to("cpu")
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

    def requires_grad(self, model, require_grad):
        for param in model.parameters():
            param.requires_grad = require_grad

    def update_dynamics_model(self, optimizer, criterion, data, epochs):
        dataset = RolloutDataset(data)
        dataloader = DataLoader(
            dataset, batch_size=self.config["train"]["batch_size"])
        for epoch in range(epochs):
            losses = 0
            for item in dataloader:
                optimizer.zero_grad()
                state, action, diff = item
                state = state.to(device=device)
                action = action.to(device=device)
                if self.config["train"]["costum_masks_for_dynamics_training"]:
                    masks = []
                    for k in range(self.config["train"]["dynamic_training_particles"]):
                        masks.append(CustomDropout(
                            dropout=self.config["train"]["dropout_sampling_dynamic"], size=self.config["train"]["hidden_size_dynamic"]))
                else:
                    masks = [nn.Dropout(p=self.config["train"]["dropout_training_dynamic"])
                             ] * self.config["train"]["dynamic_training_particles"]
                pred = torch.empty(
                    (0, self.config["train"]["output_size_dynamic"]), dtype=torch.float32, device=device)
                for mask in masks:
                    out = self.dynamics_model(state, action, mask)
                    pred = torch.cat((pred, out), dim=0)
                pred = pred.to("cpu")
                diff = diff.repeat(
                    self.config["train"]["dynamic_training_particles"], 1)
                loss = criterion(pred, diff)
                loss.backward(retain_graph=False)
                optimizer.step()
                losses += loss
        self.wandb.log(
            {"dynamics model avg loss": losses / (len(dataloader)*epochs)})
        print(f"Epochs {epochs}, avg loss {losses / (len(dataloader)*epochs)}")


class DynamicsModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(DynamicsModel, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        stack = nn.ModuleList()
        for i in range(hidden_layer):
            stack.append(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU()))
        self.mlp = nn.Sequential(*stack)
        self.final_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, action, dropout):
        x = torch.concat((x, action), dim=1)
        x = self.first_layer(x)
        x = dropout(x)
        if torch.isnan(x).any().item():
            print("nan")
        for layer in range(len(self.mlp)):
            x = self.mlp[layer](x)
            x = dropout(x)
        x = self.final_layer(x)
        return x


class PolicyModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, output_size, dropout_rate, env):
        super(PolicyModel, self).__init__()
        stack = nn.ModuleList([nn.Linear(input_size, hidden_size),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_rate)])
        for i in range(hidden_layer):
            stack.append(nn.Linear(hidden_size, hidden_size))
            stack.append(nn.LeakyReLU())
            stack.append(nn.Dropout(p=dropout_rate))
        self.mlp = nn.Sequential(*stack)
        self.final_layer = nn.Linear(hidden_size, output_size, bias=False)
        self.tanh = nn.Tanh()
        self.scale = 1
        if env == 'InvertedPendulum-v4' or env == 'SingleSwingUp':
            self.scale = 3

    def forward(self, x, eps):
        x = self.mlp(x)
        x = self.final_layer(x)
        x = self.tanh(x)
        x = x * self.scale
        x = self.epsilon_greedy(eps, x)
        return x

    def epsilon_greedy(self, eps, x):
        if torch.rand(1) < eps:
            eps_action = (torch.rand(1, device=device) - 0.5) * 2 * self.scale
            if len(x.shape) == 2:
                eps_action = torch.unsqueeze(eps_action,dim=0)
            return eps_action
        else:
            return x

    def update(self, optimizer, cost, model):
        cost = -cost
        cost.backward()
        # Gradient Norm Clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5, norm_type=2)
        optimizer.step()


class RolloutDataset(Dataset):
    def __init__(self, trajectory):
        self.trajectory = trajectory

    def __len__(self):
        return len(self.trajectory)

    def __getitem__(self, idx):
        item = self.trajectory[idx]
        return item["state"], item["action"], item["diff"]


class CustomDropout(nn.Module):
    def __init__(self, dropout, size):
        super(CustomDropout, self).__init__()
        self.mask = self.sample_mask(dropout, size)
        self.mask = torch.unsqueeze(self.mask, dim=0).to(device=device)

    def sample_mask(self, dropout, size):
        return torch.bernoulli(torch.empty(size).uniform_(1-(dropout*2), 1)) * (1/(1-dropout))

    def forward(self, x):
        return x * self.mask
