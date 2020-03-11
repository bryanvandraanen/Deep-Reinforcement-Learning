from config import *


def print_episode_results(episode_num, episode_rewards, interval_size):
    print("Episode", episode_num, "\t", "Average reward:", np.round(np.mean(
        episode_rewards[-interval_size:])), "\t", "Max reward:", np.max(episode_rewards[-interval_size:]))


def print_solved_results(episode_num, episode_rewards, proficiency_threshold):
    print("Solved after", episode_num, "episodes achieved proficiency averaging reward:",
          np.round(np.mean(episode_rewards[-proficiency_threshold:])))


def test(agent):
    agent.eval()
    test_rewards = []

    for i in range(1, TEST_EPISODES + 1):
        state = env.reset()
        state = torch.Tensor(state)

        final_reward = 0

        done = False
        while not done:
            if TEST_RENDER:
                env.render()

            action, _ = agent.get_action(state, greedy=True)
            state, reward, done, _ = env.step(action)
            state = torch.Tensor(state)

            final_reward += reward
        
        print("Test", i, "reward:", final_reward)
        test_rewards.append(final_reward)
    
    print("Completed with average reward", np.round(np.mean(test_rewards), decimals=1))


def plot_episode_rewards(episode_rewards, episodes=None, title="Episode Rewards", xlabel="Episode", ylabel="Reward"):
    global PLOT_COUNT
    plt.figure(PLOT_COUNT)
    PLOT_COUNT += 1

    if episodes == None:
        episodes = range(0, len(episode_rewards))

    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.plot(episodes, episode_rewards)


def plot_average_rewards(episode_rewards, interval, title="Average Episode Rewards"):
    average_rewards = [np.mean(episode_rewards[i:min(i+interval, len(episode_rewards))]) for i in range(0, len(episode_rewards), interval)]
    episodes = [i * interval for i in range(0, len(average_rewards))]

    plot_episode_rewards(average_rewards, episodes, title=title, ylabel="Average Reward ({} episodes)".format(interval))
