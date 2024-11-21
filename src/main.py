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


def plot_losses(losses, save_plot_dir):
    x = np.arange(1, len(losses) + 1, 1)
    fig, ax = plt.subplots()
    ax.plot(x, losses)
    ax.set(xlabel="epoch", ylabel="loss",
           title="losses evolution")
    ax.grid()
    if save_plot_dir != "":
        create_path_if_not_exist(save_plot_dir)
        fig.savefig(os.path.join(save_plot_dir, "loss_evolution.png"))
    plt.show()


def plot_score(scores, save_plot_dir):
    x = np.arange(1, len(scores) + 1, 1)
    fig, ax = plt.subplots()
    ax.plot(x, scores)
    ax.set(xlabel="episodes", ylabel="score",
           title="score evolution")
    ax.grid()
    if save_plot_dir != "":
        create_path_if_not_exist(save_plot_dir)
        fig.savefig(os.path.join(save_plot_dir, "score_evolution.png"))
    plt.show()


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


def training(agent: Agent, num_episode: int = 1000, epsilon_start: float = 1.0, epsilon_min: float = 0.01, epsilon_decay_rate: float = 0.999,seed=0, save_plot_dir:str = "", save_dir: str = ""):
    env = gym.make('LunarLander-v3')
    env.action_space.seed(seed)
    epsilon = epsilon_start
    avg_rewards_per_episode = []
    scores = []
    print("\n")
    for episode in tqdm.tqdm(range(num_episode)):
        obs, _ = env.reset()
        avg_rewards = 0
        i = 0
        score = 0
        max_steps = 1000
        # for timestep in range(max_timestep):
        for step in range(max_steps):
            action = agent.play_action(obs, epsilon)
            new_obs, reward, done, inf, _ = env.step(action)
            agent.step(obs, action, reward, new_obs, done)
            obs = new_obs
            avg_rewards += reward
            score += reward
            i += 1
            if done or step ==  max_steps-1:
                # print(f'Episode: {episode} | Steps: {step} | Total reward: {score} \
                #                     | Epsilon: {epsilon}')
                # i = timestep + 1
                break
        epsilon = max(epsilon_min, epsilon * epsilon_decay_rate)
        avg_rewards /= i
        avg_rewards_per_episode.append(avg_rewards)
        scores.append(score)
        # if (episode+1) % num_iter_before_printing == 0:
        #     print("\n episode: {}, Average reward: {}".format(episode+1, avg_rewards))
    avg_rewards_per_episode = np.array(avg_rewards_per_episode)
    plot_losses(agent.losses, save_plot_dir)
    plot_score(scores, save_plot_dir)
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
    agent = Agent(0)
    training(agent, num_episode=750, save_plot_dir="plots/", save_dir="models/")
    model_evaluation("models/")


