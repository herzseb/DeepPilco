import wandb
import yaml
import argparse
import os
import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
import time
from utils.deep_pilco import DeepPilco
from utils.environment import CartPoleSwingUp
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = 'cpu'

timestr = time.strftime("%Y%m%d-%H%M%S")
print("Model saved: ", timestr)


# TODO
# remove FIXMES
# check of BNN needs to be sampled an extra layer
# implement KL div loss
# check cuda speed in cluster


def main(config, wandb):
    # init environment
    if config["train"]["gym"] == 'InvertedDoublePendulum-v2' or config["train"]["gym"] == 'InvertedPendulum-v2':
        env = gym.make(config["train"]["gym"])
    elif config["train"]["gym"] == 'SingleSwingUp':
        env = CartPoleSwingUp()
    # seeds 
    np.random.seed(config["train"]["seed"])
    torch.manual_seed(config["train"]["seed"])
    # init polciy
    agent = DeepPilco(config, env, wandb)
    # initialise policy parameters randomly
    # agent.init_policy_parameters() # maybe add Xavier init

    optimizer_dynamics_model = optim.Adam(
        agent.dynamics_model.parameters(), lr=1e-4)
    optimizer_policy = optim.Adam(
        agent.policy.parameters(), lr=1e-6)
    last_cost = np.inf
    rollouts = []
    # 3 repeat until convergence
    while True:
        agent.requires_grad(model=agent.policy, require_grad=False)
        agent.requires_grad(model=agent.dynamics_model, require_grad=True)
        # 4 sample rollout
        rollouts.append(agent.sample_trajectory(env=env, T=config["train"]["T"]))
        if len(rollouts) > config["train"]["playback_len"]:
            rollouts = rollouts[1:]
        data = [item  for items in rollouts for item in items]
        # 5 learn dynamics model
        agent.update_dynamics_model(optimizer=optimizer_dynamics_model, criterion=nn.MSELoss(
        ), data=data, epochs=config["train"]["epochs_dynamic"])

        agent.requires_grad(model=agent.policy, require_grad=True)
        agent.requires_grad(model=agent.dynamics_model, require_grad=False)
        costs = 0
        for epoch in range(config["train"]["epochs_policy"]):
            optimizer_policy.zero_grad()
            # 6 predict trjactories fomr p(X0) to p(Xt) predict_trajectories() and 7 Evaluate policy
            trajectory, cost = agent.predict_trajectories()
            # 8 Optimize policy
            agent.policy.update(optimizer=optimizer_policy, cost=cost, model=agent.policy)
            costs += cost.item()
        avg_costs = costs / config["train"]["epochs_policy"]
        print(f'Avg policy cost {avg_costs}')
        wandb.log({"policy avg cost": avg_costs})

        if avg_costs < last_cost:
            last_cost = avg_costs
            best_model_state = deepcopy(agent.policy.state_dict())
            torch.save(best_model_state, 'policy' + timestr + '.pth')

        # if torch.abs(cost - last_cost) < config["train"]["epsilon"]:
        #     break
        #FIXME
        if avg_costs < config["train"]["epsilon"]:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process config path')
    parser.add_argument('config_file', type=str, help='name of config file')
    args = parser.parse_args()
    with open(os.path.join("configs", args.config_file), "r") as yamlfile:
        config_yaml = yaml.load(yamlfile, Loader=yaml.FullLoader)
    config = {}
    for item in config_yaml:
        key = next(iter(item))
        config[key] = item[key]
    print(config)
    wandb.init(project=config["log"]["project"], config=config)
    main(config, wandb)
    wandb.finish()
