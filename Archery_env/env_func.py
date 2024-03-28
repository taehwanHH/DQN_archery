import numpy as np
from math import pi, cos, sin


def polar2cartesian(v):
    r = v[0]
    theta = v[1]
    return [r*cos(theta), r*sin(theta)]


def cartesian2polar(v):
    x = v[0]
    y = v[1]
    return [np.linalg.norm(v, ord=2), np.arctan2(y, x)]


def wind_dir_trans(state):
    [x, y] = polar2cartesian(state)
    # wind velocity transition probability
    dv_x = np.random.uniform(low=-2, high=2)
    dv_y = np.random.uniform(low=-2, high=2)

    next_wind = [x+dv_x, y+dv_y]

    return np.array(next_wind)


def conditional_pmf(idx):
    # humidity transition probability mass function
    pmf = {}

    if idx == 0:
        pmf[0] = 2/3
        pmf[1] = 1/3
    elif idx == 4:
        pmf[3] = 1/3
        pmf[4] = 2/3
    else:
        pmf[idx-1] = 1/3
        pmf[idx] = 1/3
        pmf[idx+1] = 1/3

    return pmf


def humidity_trans(state, h_set):
    # humidity transition probability

    h_set = np.array(h_set)
    h_origin = state[2]
    idx = np.where(h_set == h_origin)[0][0]

    pmf = conditional_pmf(idx)

    indices = list(pmf.keys())
    prob = list(pmf.values())

    selected_index = np.random.choice(indices, p=prob)

    return h_set[selected_index]


def wind_coeff(humidity):
    return -0.005*humidity+1

# def v_polar2cartesian(polar_v):
#     r = polar_v[:, 0]
#     theta = polar_v[:, 1]
#
#     x = r * np.cos(theta)
#     y = r * np.sin(theta)
#
#     return np.column_stack((x, y))
#
#
# def optimal_policy(state, cartesian_action_set, action_set):
#     eff_wind_vector = wind_coeff(state[2]) * np.array(polar2cartesian(state[0:2]))  # effective wind vector
#     optimal_action = -1 * eff_wind_vector
#
#     distances = np.sqrt(np.sum((cartesian_action_set - optimal_action)**2, axis=1))
#
#     closest_index = np.argmin(distances)
#
#     return action_set[closest_index]


def get_score(state, action):
    eff_wind_vector = wind_coeff(state[2]) * np.array(polar2cartesian(state[0:2]))  # effective wind vector
    aiming_vector = polar2cartesian(action)
    reach_point = np.array(eff_wind_vector + aiming_vector)

    distance = np.linalg.norm(reach_point, ord=2)
    score = 11 - distance
    score = int(score)
    score = score if score > 0 else 0

    return distance, score