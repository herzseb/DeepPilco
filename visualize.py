import argparse
import gym
import torch
import argparse
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.deep_pilco import DeepPilco

def visualize(config):
    env = gym.make('InvertedDoublePendulum-v4', render_mode="rgb_array")
    env.action_space.seed(42)
    env.reset()
    agent = DeepPilco(config, env, None)
    agent.policy.load_state_dict(torch.load("policy20230615-120942.pth"))
    agent.policy.eval()
    observation, info = env.reset(seed=42)

    with torch.no_grad():
        for _ in range(1000):
            observation = torch.tensor(observation).to(dtype=torch.float32)
            observation, reward, terminated, truncated, info = env.step(agent.policy(observation))
            #print(observation[1]**2 + observation[2]**2)
            # print(observation[1],observation[2],observation[3],  observation[4], np.abs(observation[1]) +  observation[4])
            d = np.sqrt((observation[3]*1 + observation[4]*1 - (1+1))** 2 + 0.1 * (observation[1]*1 + observation[2]*1)**2)
            print(d)
            # print(np.sin((np.arcsin(observation[2]) + 3.14159) - (1.57 + np.arcsin(observation[1]))))
            # print(observation[1:5])
            img = env.render()
            plt.imshow(img)
            plt.show()
            # if terminated or truncated:
            #     observation, info = env.reset()

    env.close()


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
    visualize(config)
