import torch
import gym

import numpy as np
from MLP import MultiLayerPerceptron as MLP
from DQN import DQN, prepare_training_inputs
from memory import ReplayMemory
from train_utils import to_tensor
from env.Archery import ArcheryEnv
import matplotlib.pyplot as plt
from scipy.io import savemat

lr = 1e-4 * 5
batch_size = 256
gamma = 0.98
memory_size = 50000
total_eps = 10000
eps_max = 0.3
eps_min = 0.01
sampling_only_until = 300
target_update_interval = 4

env = ArcheryEnv()

s_dim = env.state_dim
a_dim = env.action_space.n

qnet = MLP(s_dim, a_dim, num_neurons=[64, 64, 64])
qnet_target = MLP(s_dim, a_dim, num_neurons=[64, 64, 64])

# initialize target network same as the main network.
qnet_target.load_state_dict(qnet.state_dict())
agent = DQN(s_dim, 1, qnet=qnet, qnet_target=qnet_target, lr=lr, gamma=gamma, epsilon=1.0)

memory = ReplayMemory(memory_size)

print_every = 100

cum_score_list =[]
for n_epi in range(total_eps):
    # epsilon scheduling
    # slowly decaying_epsilon
    epsilon = max(eps_min, eps_max - eps_min * (n_epi /150))
    agent.epsilon = torch.tensor(epsilon)
    s = env.reset()

    while True:
        s = to_tensor(s, size=(3,))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a)

        experience = (torch.tensor(s).view(1,3),
                      torch.tensor([a]).view(1,1),
                      torch.tensor([r]).view(1, 1),
                      torch.tensor(ns).view(1,3),
                      torch.tensor([done]).view(1,1 ))
        memory.push(experience)
        env.state = ns
        s = env.state
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
        msg = (n_epi, env.cum_score, epsilon)
        print("Episode : {:4.0f} | Cumulative score : {:4.0f} | Epsilon : {:.3f}".format(*msg))

    cum_score_list.append(env.cum_score)

epi = np.arange(total_eps)
plt.plot(epi, cum_score_list)
plt.show()

savemat("data.mat",{'sim_res' : cum_score_list})