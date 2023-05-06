import wandb
import yaml
import argparse
import os
import gym
import numpy as np
import torch.optim as optim
import torch.nn as nn
from utils.deep_pilco import DeepPilco

# TODO
# remove FIXMES
# debug through all steps
# check of BNN needs to be sampled an extra layer
# of many trajectories are used to update dynamics model 
# implement double inverted gym
# implement KL div loss
# continues loss
# use cuda

def main(config):
    # init environment
    env = gym.make('CartPole-v0')
    # init polciy
    agent = DeepPilco()
    # initialise policy parameters randomly
    agent.init_policy_parameters()

    optimizer_dynamics_model = optim.SGD(agent.dynamics_model.parameters(), lr=0.001, momentum=0.9)
    optimizer_policy = optim.SGD(agent.policy.parameters(), lr=0.001, momentum=0.9)
    last_cost = np.inf
    # 3 repeat until convergence
    while True:
        agent.requires_grad(model=agent.policy, require_grad=False)
        agent.requires_grad(model=agent.dynamics_model, require_grad=True)
        optimizer_dynamics_model.zero_grad()
        # 4 sample rollout
        rollout = agent.sample_trajectory(env=env, T=config["train"]["T"])
        # 5 learn dynamics model
        agent.dynamics_model.update(optimizer=optimizer_dynamics_model, criterion=nn.MSELoss(), data=rollout)

        agent.requires_grad(model=agent.policy, require_grad=True)
        agent.requires_grad(model=agent.dynamics_model, require_grad=False)
        optimizer_policy.zero_grad()
        # 6 predict trjactories fomr p(X0) to p(Xt) predict_trajectories()
        trajectory = agent.predict_trajectories()
        # 7 Evaluate policy
        cost = agent.evaluate_policy()
        # 8 Optimize policy
        agent.policy.update(optimizer=optimizer_policy, criterion=nn.MSELoss(), cost=cost)

        if np.abs(cost - last_cost) < config["train"]["epsilon"]:
            break
        last_cost = cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process config path')
    parser.add_argument('config_file', type=str, help='name of config file')
    args = parser.parse_args()
    with open(os.path.join("configs", args.config_file), "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)[0]
    print(config)
    wandb.init(project=config["log"]["project"], config=config)
    main(config)
    wandb.finish()
