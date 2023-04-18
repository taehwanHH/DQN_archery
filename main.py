import torch
import gym

import numpy as np
from MLP import MultiLayerPerceptron as MLP
from DQN import DQN, prepare_training_inputs
from memory import ReplayMemory
from train_utils import to_tensor
from env.temp_env import CommunicationEnv
import matplotlib.pyplot as plt
lr = 1e-4 * 5
batch_size = 64
gamma = 0.95
memory_size = 50000
total_eps = 100
eps_max = 0.6
eps_min = 0.1
sampling_only_until = 100
target_update_interval = 5

env = CommunicationEnv()
s_dim = env.state_space.shape[0]
a_dim = env.action_space.n


qnet = MLP(s_dim, a_dim, num_neurons=[128, 256])
qnet_target = MLP(s_dim, a_dim, num_neurons=[128, 256])

# initialize target network same as the main network.
qnet_target.load_state_dict(qnet.state_dict())
agent = DQN(s_dim, a_dim, qnet=qnet, qnet_target=qnet_target, lr=lr, gamma=gamma, epsilon=1.0)

memory = ReplayMemory(memory_size)

print_every = 1

cum_r_list =[]
final_eta_list =[]
for n_epi in range(total_eps):
    # epsilon scheduling
    # slowly decaying_epsilon
    epsilon = max(eps_min, eps_max - eps_min * (n_epi /20))
    agent.epsilon = torch.tensor(epsilon)
    s = env.reset()
    cum_r = 0.0

    while True:
        s = to_tensor(s, size=(1,1))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a)

        experience = (s,
                      torch.tensor(a).view(1, 1),
                      torch.tensor(r).view(1, 1),
                      torch.tensor(ns).view(1, 1),
                      torch.tensor(done).view(1, 1))
        memory.push(experience)
        print(experience)
        s = ns
        cum_r += r
        if done:
            break

    if len(memory) >= sampling_only_until:
        # train agent
        sampled_exps = memory.sample(batch_size)
        sampled_exps = prepare_training_inputs(sampled_exps)
        agent.update(*sampled_exps)

    if n_epi % target_update_interval == 0:
        qnet_target.load_state_dict(qnet.state_dict())

    if n_epi % print_every == 0:
        msg = (n_epi, cum_r, epsilon)
        print("Episode : {:4.0f} | Cumulative Reward : {:4.0f} | Epsilon : {:.3f}".format(*msg))

    cum_r_list.append(cum_r)
    final_eta_list.append(s)

epi = np.arange(total_eps)
plt.plot(epi, cum_r_list)
plt.plot(epi, final_eta_list)

plt.show()