import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import tqdm

from agent import Agent


def create_path_if_not_exist(path: str) -> None:
    """
    Create if exists missing directory in a path to a file
    :param path: A file path
    :return:
    """
    dir_path, file = os.path.split(path)
    # Create missing directory in the path
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def plot_reward(num_episode, avg_reward, title, save_plot_dir):
    x = np.arange(1, num_episode+1, 1)
    fig, ax = plt.subplots()
    ax.plot(x, avg_reward)
    ax.set(xlabel="episode", ylabel="Avg Reward",
           title=title)
    ax.grid()
    if save_plot_dir != "":
        create_path_if_not_exist(save_plot_dir)
        fig.savefig(os.path.join(save_plot_dir, "avg_reward_per_episode.png"))
    plt.show()


def training(agent: Agent, num_episode: int = 1000, epsilon_start: float = 1.0, epsilon_min: float = 0.01, epsilon_decay_rate: float = 0.997, save_plot_dir:str = "", save_dir: str = ""):
    env = gym.make('LunarLander-v3')
    epsilon = epsilon_start
    avg_rewards_per_episode = []
    # num_iter_before_printing = num_episode // 20
    print("/n")
    for episode in tqdm.tqdm(range(num_episode)):
        obs, _ = env.reset()
        avg_rewards = 0
        i = 0
        # for timestep in range(max_timestep):
        while True:
            action = agent.play_action(obs, epsilon)
            new_obs, reward, done, inf, _ = env.step(action)
            agent.step(obs, action, reward, new_obs, done)
            obs = new_obs
            avg_rewards += reward
            i += 1
            if done:
                # i = timestep + 1
                break
        avg_rewards /= i
        avg_rewards_per_episode.append(avg_rewards)
        epsilon = max(epsilon_min, epsilon * epsilon_decay_rate)
        # if (episode+1) % num_iter_before_printing == 0:
        #     print("\n episode: {}, Average reward: {}".format(episode+1, avg_rewards))
    avg_rewards_per_episode = np.array(avg_rewards_per_episode)
    plot_reward(num_episode, avg_rewards_per_episode, "Avg Reward per episode", save_plot_dir)
    if save_dir != "":
        # save_path = "./models"
        create_path_if_not_exist(save_dir)
        agent.save_weights(save_dir)
    env.close()


def model_evaluation(load_path_dir):
    env = gym.make('LunarLander-v3', render_mode='human')
    # env.metadata["render_fps"] = 180
    obs, _ = env.reset()
    model = Agent.load_ai_model_weights(load_path_dir)
    while True:
        obs = obs.reshape(1, -1)
        action = np.argmax(model.predict(obs, verbose=0), axis=1)[0]
        new_obs, reward, done, inf, _ = env.step(action)
        obs = new_obs
        if done:
            break
    env.close()


if __name__ == "__main__":
    agent = Agent(42)
    training(agent, num_episode=1000, save_plot_dir="plots/", save_dir="models/")
    model_evaluation("models/")
