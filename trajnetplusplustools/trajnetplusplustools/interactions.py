""" Categorizes the Interaction """

import numpy as np

from . import kalman
from . import metrics

#######################################
## Helper Functions for interactions ##
#######################################

def compute_velocity_interaction(path, neigh_path, obs_len=9, stride=3):
    ## Computes the angle between velocity of neighbours and velocity of pp

    prim_vel = path[obs_len:] - path[obs_len-stride:-stride]
    theta1 = np.arctan2(prim_vel[:, 1], prim_vel[:, 0])
    neigh_vel = neigh_path[obs_len:] - neigh_path[obs_len-stride:-stride]
    vel_interaction = np.zeros(neigh_vel.shape[0:2])
    sign_interaction = np.zeros(neigh_vel.shape[0:2])

    for n in range(neigh_vel.shape[1]):
        theta2 = np.arctan2(neigh_vel[:, n, 1], neigh_vel[:, n, 0])
        theta_diff = (theta2 - theta1) * 180 / np.pi
        theta_diff = theta_diff % 360
        theta_sign = theta_diff > 180
        sign_interaction[:, n] = theta_sign
        vel_interaction[:, n] = theta_diff
    return vel_interaction, sign_interaction


def compute_theta_interaction(path, neigh_path, obs_len=9, stride=3):
    ## Computes the angle between line joining pp to neighbours and velocity of pp

    prim_vel = path[obs_len:] - path[obs_len-stride:-stride]
    theta1 = np.arctan2(prim_vel[:, 1], prim_vel[:, 0])
    rel_dist = neigh_path[obs_len:] - path[obs_len:][:, np.newaxis, :]
    theta_interaction = np.zeros(rel_dist.shape[0:2])
    sign_interaction = np.zeros(rel_dist.shape[0:2])

    for n in range(rel_dist.shape[1]):
        theta2 = np.arctan2(rel_dist[:, n, 1], rel_dist[:, n, 0])
        theta_diff = (theta2 - theta1) * 180 / np.pi
        theta_diff = theta_diff % 360
        theta_sign = theta_diff > 180
        sign_interaction[:, n] = theta_sign
        theta_interaction[:, n] = theta_diff
    return theta_interaction, sign_interaction

def compute_dist_rel(path, neigh_path, obs_len=9):
    ## Distance between pp and neighbour

    dist_rel = np.linalg.norm((neigh_path[obs_len:] - path[obs_len:][:, np.newaxis, :]), axis=2)
    return dist_rel


def compute_interaction(theta_rel_orig, dist_rel, angle, dist_thresh, angle_range):
    ## Interaction is defined as
    ## 1. distance < threshold and
    ## 2. angle between velocity of pp and line joining pp to neighbours

    theta_rel = np.copy(theta_rel_orig)
    angle_low = (angle - angle_range)
    angle_high = (angle + angle_range)
    if (angle - angle_range) < 0:
        theta_rel[np.where(theta_rel > 180)] = theta_rel[np.where(theta_rel > 180)] - 360
    if (angle + angle_range) > 360:
        raise ValueError
    interaction_matrix = (angle_low < theta_rel) & (theta_rel <= angle_high) \
                         & (dist_rel < dist_thresh) & (theta_rel < 500) == 1
    return interaction_matrix

def interaction_length(interaction_matrix, length=1):
    interaction_sum = np.sum(interaction_matrix, axis=0)
    return interaction_sum >= length

def check_interaction(rows, pos_range=15, dist_thresh=5, choice='pos', \
                      pos_angle=0, vel_angle=0, vel_range=15, output='matrix', obs_len=9):

    path = rows[:, 0]
    neigh_path = rows[:, 1:]
    theta_interaction, _ = compute_theta_interaction(path, neigh_path, obs_len)
    vel_interaction, _ = compute_velocity_interaction(path, neigh_path, obs_len)
    dist_rel = compute_dist_rel(path, neigh_path, obs_len)

    ## str choice
    if choice == 'pos':
        interaction_matrix = compute_interaction(theta_interaction, dist_rel, \
                                                 pos_angle, dist_thresh, pos_range)
        chosen_interaction = theta_interaction

    elif choice == 'vel':
        interaction_matrix = compute_interaction(vel_interaction, dist_rel, \
                                                 vel_angle, dist_thresh, vel_range)
        chosen_interaction = vel_interaction

    elif choice == 'bothpos':
        pos_matrix = compute_interaction(theta_interaction, dist_rel, \
                                         pos_angle, dist_thresh, pos_range)
        vel_matrix = compute_interaction(vel_interaction, dist_rel, \
                                         vel_angle, dist_thresh, vel_range)
        interaction_matrix = pos_matrix & vel_matrix
        chosen_interaction = theta_interaction

    elif choice == 'bothvel':
        pos_matrix = compute_interaction(theta_interaction, dist_rel, \
                                         pos_angle, dist_thresh, pos_range)
        vel_matrix = compute_interaction(vel_interaction, dist_rel, \
                                         vel_angle, dist_thresh, vel_range)
        interaction_matrix = pos_matrix & vel_matrix
        chosen_interaction = vel_interaction
    else:
        raise NotImplementedError

    chosen_true = chosen_interaction[interaction_matrix]
    dist_true = dist_rel[interaction_matrix]

    if output == 'matrix':
        return interaction_matrix
    if output == 'all':
        return interaction_matrix, chosen_true, dist_true

    return np.any(interaction_matrix)

def check_group(rows, dist_thresh=0.8, std_thresh=0.2, obs_len=9):
    ## Identify Groups
    ## dist_thresh: Distance threshold to be withinin a group
    ## std_thresh: Std deviation threshold for variation of distance

    path = rows[:, 0]
    neigh_path = rows[:, 1:]

    ## Horizontal Position
    interaction_matrix_1 = check_interaction(rows, pos_angle=90, pos_range=45, obs_len=obs_len)
    interaction_matrix_2 = check_interaction(rows, pos_angle=270, pos_range=45, obs_len=obs_len)
    neighs_side = np.any(interaction_matrix_1, axis=0) | np.any(interaction_matrix_2, axis=0)

    ## Distance Maintain
    dist_rel = np.linalg.norm((neigh_path - path[:, np.newaxis, :]), axis=2)
    mean_dist = np.mean(dist_rel, axis=0)
    std_dist = np.std(dist_rel, axis=0)

    group_matrix = (mean_dist < dist_thresh) & (std_dist < std_thresh) & neighs_side

    return group_matrix

#####################################
## Functions for Interaction types ##
#####################################

## Type 2
def non_linear(scene, obs_len=9, pred_len=12):
    primary_prediction, _ = kalman.predict(scene, obs_len, pred_len)[0]
    score = metrics.final_l2(scene[0], primary_prediction)
    return score > 0.5, primary_prediction

## Type 3a
def leader_follower(rows, pos_range=15, dist_thresh=5, obs_len=9):
    """ Identifying Leader Follower Behavior """
    interaction_matrix = check_interaction(rows, pos_range=pos_range, dist_thresh=dist_thresh,
                                           choice='bothpos', obs_len=obs_len)
    interaction_index = interaction_length(interaction_matrix, length=5)
    return interaction_index

## Type 3b
def collision_avoidance(rows, pos_range=15, dist_thresh=5, obs_len=9):
    """ Identifying Collision Avoidance Behavior """
    interaction_matrix = check_interaction(rows, pos_range=pos_range, dist_thresh=dist_thresh, \
                                           choice='bothpos', vel_angle=180, obs_len=obs_len)
    interaction_index = interaction_length(interaction_matrix, length=1)
    return interaction_index

## Type 3c
def group(rows, dist_thresh=0.8, std_thresh=0.2, obs_len=9):
    """ Identifying Group Behavior """
    # print("HERE")
    interaction_index = check_group(rows, dist_thresh, std_thresh, obs_len)
    return interaction_index

## Get Type
def get_interaction_type(rows, pos_range=15, dist_thresh=5, obs_len=9):
    interaction_type = []
    if np.any(leader_follower(rows, pos_range, dist_thresh, obs_len)):
        interaction_type.append(1)
    if np.any(collision_avoidance(rows, pos_range, dist_thresh, obs_len)):
        interaction_type.append(2)
    if np.any(group(rows, obs_len=obs_len)):
        interaction_type.append(3)
    if interaction_type == []:
        interaction_type.append(4)
    return interaction_type
