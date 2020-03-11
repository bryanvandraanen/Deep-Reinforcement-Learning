from config import *
from util import *


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, dropout):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.softmax(x, dim=0)

        return x

    def get_action(self, state, greedy=False):
        probs = self(state)

        distribution = Categorical(probs)
        action = distribution.probs.max(-1)[1] if greedy else distribution.sample()

        log_prob = distribution.log_prob(action)

        return action.item(), log_prob.squeeze(0)


def reinforce_update(optimizer, policy, rewards, log_probs):
    returns = calculate_returns(rewards)

    log_probs = np.array(log_probs)
    returns = np.array(returns)

    policy_loss = -log_probs * returns

    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss.tolist()).sum()
    policy_loss.backward()
    optimizer.step()


def calculate_returns(rewards, normalize=True):
    returns = []

    for i in range(len(rewards)):
        reward_return = 0
        for i, r in enumerate(rewards[i:]):
            reward_return += GAMMA ** i * r

        returns.append(reward_return)

    # Normalize returns to increase training stability/reduce variance (average provides baseline)
    returns = torch.Tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + VERY_SMALL_NUMBER)

    return returns


def REINFORCE(policy, optimizer):
    episode_rewards = []

    for episode in range(1, EPISODES):
        state = env.reset()
        state = torch.Tensor(state)

        log_probs = []
        rewards = []

        for step in range(1, MAX_STEPS):
            if TRAIN_RENDER:
                env.render()

            action, log_prob = policy.get_action(state)

            new_state, reward, done, _ = env.step(action)
            new_state = torch.Tensor(new_state)

            log_probs.append(log_prob)
            rewards.append(reward)

            state = new_state

            if done:
                break

        episode_reward = np.sum(rewards)
        episode_rewards.append(episode_reward)

        reinforce_update(optimizer, policy, rewards, log_probs)

        if episode % LOG_INTERVAL == 0:
            print_episode_results(episode, episode_rewards, LOG_INTERVAL)

        if np.mean(episode_rewards[-PROFICIENCY:]) > env.spec.reward_threshold:
            print_solved_results(episode, episode_rewards, PROFICIENCY)
            break

    return episode_rewards


if __name__ == '__main__':
    reinforce_agent = Actor(state_dim, action_dim, HIDDEN_DIM, DROPOUT)
    reinforce_optimizer = optim.Adam(reinforce_agent.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    episode_rewards = REINFORCE(reinforce_agent, reinforce_optimizer)

    print("Running test for REINFORCE using optimal policy now for", TEST_EPISODES, "episodes")
    test(reinforce_agent)

    env.close()

    plot_episode_rewards(episode_rewards, title="REINFORCE Training Rewards")
    plot_average_rewards(episode_rewards, LOG_INTERVAL, title="REINFORCE Average Training Rewards")

    plt.show()
