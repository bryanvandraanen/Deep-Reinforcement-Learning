from config import *
from reinforce import Actor
from util import *


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return x.detach().item()


def a2c_update(optimizer, rewards, log_probs, state_values):
    q_values = calculate_q_values(rewards)

    log_probs = torch.stack(log_probs)
    state_values = torch.Tensor(state_values)

    advantages = q_values - state_values

    value_loss = (advantages ** 2).mean()
    policy_loss = (-log_probs * advantages).mean()

    optimizer.zero_grad()
    (policy_loss + value_loss).backward()
    optimizer.step()


def calculate_q_values(rewards):
    q_value = 0
    q_values = [0] * len(rewards)
    for i in range(len(rewards) - 1, -1, -1):
        q_value = rewards[i] + GAMMA * q_value
        q_values[i] = q_value

    return torch.Tensor(q_values)


def A2C(actor, critic, optimizer):
    episode_rewards = []

    for episode in range(EPISODES):
        state = env.reset()
        state = torch.Tensor(state)

        log_probs = []
        rewards = []
        state_values = []

        for step in range(MAX_STEPS):
            if TRAIN_RENDER:
                env.render()

            value = critic(state)
            action, log_prob = actor.get_action(state)

            new_state, reward, done, _ = env.step(action)
            new_state = torch.Tensor(new_state)

            rewards.append(reward)
            state_values.append(value)
            log_probs.append(log_prob)

            state = new_state

            if done:
                break

        episode_reward = np.sum(rewards)
        episode_rewards.append(episode_reward)

        a2c_update(optimizer, rewards, log_probs, state_values)

        if episode % LOG_INTERVAL == 0:
            print_episode_results(episode, episode_rewards, LOG_INTERVAL)

        if np.mean(episode_rewards[-PROFICIENCY:]) > env.spec.reward_threshold:
            print_solved_results(episode, episode_rewards, PROFICIENCY)
            break

    return episode_rewards


if __name__ == '__main__':
    a2c_agent = Actor(state_dim, action_dim, HIDDEN_DIM, DROPOUT)
    a2c_critic = Critic(state_dim, HIDDEN_DIM)

    params = list(a2c_critic.parameters()) + list(a2c_agent.parameters())
    a2c_optimizer = optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    episode_rewards = A2C(a2c_agent, a2c_critic, a2c_optimizer)

    print("Running test for A2C using optimal policy now for", TEST_EPISODES, "episodes")
    test(a2c_agent)

    env.close()

    plot_episode_rewards(episode_rewards, title="A2C Training Rewards")
    plot_average_rewards(episode_rewards, LOG_INTERVAL, title="A2C Average Training Rewards")

    plt.show()
