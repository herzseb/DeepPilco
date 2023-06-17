import argparse
import gym
import torch
import argparse
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.deep_pilco import DeepPilco
# from utils.environment import CustomGym, ExtendedCartPoleEnv, CartPoleSwingUp
from utils.environment import CartPoleSwingUp
from gym.wrappers.time_limit import TimeLimit

def visualize(config):
    max_episode_steps = 200
    # env = ExtendedCartPoleEnv(render_mode="rgb_array")
    env = CartPoleSwingUp()
    # env = TimeLimit(env, max_episode_steps)
    # env = CustomGym('InvertedPendulum-v4')
    #env = gym.make('InvertedPendulum-v4', render_mode="rgb_array")
    #env.reset_model(np.array([1.,0.,0.,0.]))
    agent = DeepPilco(config, env, None)
    agent.policy.load_state_dict(torch.load("policy20230616-175214.pth"))
    agent.policy.eval()
    #observation = env.reset_model(np.array([0.,np.pi,0.,0.]))
    observation = env.reset()

    with torch.no_grad():
        for _ in range(1000):
            observation = torch.tensor(observation).to(dtype=torch.float32)
            action = agent.policy(observation, eps=0)
            print(action)
            observation, _, _ , _ = env.step(action)
            print(np.sqrt((1 - np.cos(observation[2]))** 2 + 0.1 * (np.sin(observation[2]))**2))
            #print(observation[1]**2 + observation[2]**2)
            # print(observation[1],observation[2],observation[3],  observation[4], np.abs(observation[1]) +  observation[4])
            # d = np.sqrt((observation[3]*1 + observation[4]*1 - (1+1))** 2 + 0.1 * (observation[1]*1 + observation[2]*1)**2)
            # print(d)
            # print(np.sin((np.arcsin(observation[2]) + 3.14159) - (1.57 + np.arcsin(observation[1]))))
            # print(observation[1:5])
            #print(np.cos(observation[1]))
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
