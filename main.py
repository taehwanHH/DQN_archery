import torch
import os
from MLP import MultiLayerPerceptron as MLP
from DQN import DQN, prepare_training_inputs
from memory import ReplayMemory
from train_utils import to_tensor
from Archery_env.Archery import ArcheryEnv


import matplotlib.pyplot as plt
from scipy.io import savemat
from collections import deque
from param import Hyper_Param
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'

# Hyperparameters
lr = Hyper_Param['learning_rate']
batch_size = Hyper_Param['batch_size']
gamma = Hyper_Param['discount_factor']
memory_size = Hyper_Param['memory_size']
total_eps = Hyper_Param['num_episode']
epsilon = Hyper_Param['epsilon']
epsilon_min = Hyper_Param['epsilon_min']
eps_decay_rate = Hyper_Param['epsilon_decay_rate']
sampling_only_until = Hyper_Param['train_start']
target_update_interval = Hyper_Param['target_update_interval']
print_every = Hyper_Param['print_every']
window_size = Hyper_Param['window_size']
step_max = Hyper_Param['step_max']

# List storing the results
score_avg = deque(maxlen=window_size)
cum_score_list = []
score_avg_value = []
# cum_rand_score_list = []
# optimal_score_avg = deque(maxlen=window_size)
# optimal_score_avg_value = []
epi = []

# Create Environment
env = ArcheryEnv()
s_dim = env.state_dim
a_dim = env.action_space.n


qnet = MLP(s_dim, a_dim, num_neurons=Hyper_Param['num_neurons'])
qnet_target = MLP(s_dim, a_dim, num_neurons=Hyper_Param['num_neurons'])

# initialize target network same as the main network.
qnet_target.load_state_dict(qnet.state_dict())
agent = DQN(s_dim, a_dim, qnet=qnet, qnet_target=qnet_target, lr=lr, gamma=gamma, epsilon=1.0)

memory = ReplayMemory(memory_size)

# Episode start
for n_epi in range(total_eps):
    # epsilon scheduling
    # slowly decaying_epsilon
    agent.epsilon = torch.tensor(epsilon)
    s = env.reset()

    epi.append(n_epi)
    while True:
        s = to_tensor(s, size=(3,))
        a = agent.get_action(s)
        ns, r, done, info = env.step(a)

        experience = (torch.tensor(s).view(1,3),
                      torch.tensor([a]).view(1,1),
                      torch.tensor([r]).view(1, 1),
                      torch.tensor(ns).view(1,3),
                      torch.tensor([done]).view(1,1))
        memory.push(experience)
        env.state = ns
        s = env.state
        if done:
            break

    cum_score = env.cum_score/step_max
    score_avg.append(cum_score)
    cum_score_list.append(cum_score)
    # optimal_score_avg.append(env.cum_optimal_score/step_max)

    if len(memory) >= sampling_only_until:
        # train agent
        epsilon = max(epsilon_min, epsilon*eps_decay_rate)

        sampled_exps = memory.sample(batch_size)
        sampled_exps = prepare_training_inputs(sampled_exps)
        agent.update(*sampled_exps)

    if n_epi % target_update_interval == 0:
        qnet_target.load_state_dict(qnet.state_dict())

    if len(score_avg) == window_size:
        score_avg_value.append(sum(score_avg) / window_size)
        # cum_rand_score_list.append(env.cum_rand_score / step_max)
        # optimal_score_avg_value.append(sum(optimal_score_avg) / window_size)

    else:
        score_avg_value.append(sum(score_avg) / len(score_avg))
        # optimal_score_avg_value.append(sum(optimal_score_avg) / len(optimal_score_avg))

    if n_epi % print_every == 0:
        msg = (n_epi, cum_score, epsilon)
        print("Episode : {:4.0f} | Cumulative score : {:.2f} | Epsilon : {:.3f}".format(*msg))
        plt.xlim(0, total_eps)
        plt.ylim(0, 10)
        plt.plot(epi, cum_score_list, color='black')
        plt.plot(epi, score_avg_value, color='red')
        # plt.plot(epi, optimal_score_avg_value, color='blue')
        # plt.plot(epi, cum_rand_score_list, color='blue')
        # plt.plot(epi, cum_optimal_score_list, color='green')
        plt.xlabel('Episode', labelpad=5)
        plt.ylabel('Average score', labelpad=5)
        plt.grid(True)
        plt.pause(0.0001)
        plt.close()


# Base directory path creation
base_directory = os.path.join(Hyper_Param['today'])

# Subdirectory index calculation
if not os.path.exists(base_directory):
    os.makedirs(base_directory)
    index = 1
else:
    existing_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    indices = [int(d) for d in existing_dirs if d.isdigit()]
    index = max(indices) + 1 if indices else 1

# Subdirectory creation
sub_directory = os.path.join(base_directory, str(index))
os.makedirs(sub_directory)

# Store plt in Subdirectory
plt.xlim(0, total_eps)
plt.ylim(0, 10)
plt.plot(epi, cum_score_list, color='black')
plt.plot(epi, score_avg_value, color='red')
# plt.plot(epi, optimal_score_avg_value, color='blue')
# plt.plot(epi, cum_rand_score_list, color='blue')
# plt.plot(epi, cum_optimal_score_list, '--g')
plt.xlabel('Episode', labelpad=5)
plt.ylabel('Average score', labelpad=5)
plt.grid(True)
plt.savefig(os.path.join(sub_directory, f"plot_{index}.png"))

# Store Hyperparameters in txt file
with open(os.path.join(sub_directory, 'Hyper_Param.txt'), 'w') as file:
    for key, value in Hyper_Param.items():
        file.write(f"{key}: {value}\n")

# Store score data (matlab data file)
savemat(os.path.join(sub_directory, 'data.mat'),{'sim_res': cum_score_list})
# savemat(os.path.join(sub_directory, 'data.mat'),{'sim_res': cum_score_list,'sim_optimal': optimal_score_avg_value})
# savemat(os.path.join(sub_directory, 'data.mat'), {'sim_res': cum_score_list,'sim_rand_res': cum_rand_score_list,
#                                                   'sim_optimal_res': cum_optimal_score_list})
