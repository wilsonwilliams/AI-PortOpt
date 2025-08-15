import os

from environment import PortfolioEnv

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


if __name__ == "__main__":
    ### RUN CODE TO TRAIN MODEL ###
    log_returns = pd.read_csv(os.path.join("data", "train_returns.csv"))
    vols = pd.read_csv(os.path.join("data", "train_vols.csv"))

    env = make_vec_env(lambda: PortfolioEnv(log_returns, vols), n_envs=1)

    print('=' * 20 + "  Testing Environment  " + '=' * 20)
    episodes = 5
    for episode in range(1, episodes+1):
        obs = env.reset()
        done = False
        
        while not done:
            action = env.action_space.sample().reshape(1, 15)
            obs, reward, done, info = env.step(action)
        print("Episode: {} | Portfolio Value: {}".format(episode, round(info[0]["PortValue"], 2)))

    print('=' * 20 + "  Environment Testing Complete  " + '=' * 20)
    print("\n\n" + '=' * 20 + "  Training Model  " + '=' * 20)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=os.path.join("logs"))
    model.learn(total_timesteps=100_000)
    model.save(os.path.join("models", "PPO_100K_Model"))
    
    print("\n\n" + '=' * 20 + "  Training Complete  " + '=' * 20 + "\n")
