import math
import random
from csv import writer
import numpy
import os
import pdb
from contextlib import contextmanager

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math
import pdb


def random_rotation(xy, goals=None):
    theta = random.random() * 2.0 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)
    r = numpy.array([[ct, st], [-st, ct]])
    if goals is None:
        return numpy.einsum('ptc,ci->pti', xy, r)
    return numpy.einsum('ptc,ci->pti', xy, r), numpy.einsum('tc,ci->ti', goals, r)

def shift(xy, center):
    # theta = random.random() * 2.0 * math.pi
    xy = xy - center[numpy.newaxis, numpy.newaxis, :]
    return xy

def theta_rotation(xy, theta):
    # theta = random.random() * 2.0 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)

    r = numpy.array([[ct, st], [-st, ct]])
    return numpy.einsum('ptc,ci->pti', xy, r)

def center_scene(xy, obs_length=9, ped_id=0, goals=None):
    if goals is not None:
        goals = goals[numpy.newaxis, :, :]
    ## Center
    center = xy[obs_length-1, ped_id] ## Last Observation
    xy = shift(xy, center)
    if goals is not None:
        goals = shift(goals, center)

    ## Rotate
    last_obs = xy[obs_length-1, ped_id]
    second_last_obs = xy[obs_length-2, ped_id]
    diff = numpy.array([last_obs[0] - second_last_obs[0], last_obs[1] - second_last_obs[1]])
    thet = numpy.arctan2(diff[1], diff[0])
    rotation = -thet + numpy.pi/2
    xy = theta_rotation(xy, rotation)
    if goals is not None:
        goals = theta_rotation(goals, rotation)
        return xy, rotation, center, goals[0]
    return xy, rotation, center

def visualize_scene(scene, goal=None):
    for t in range(scene.shape[1]):
        path = scene[:, t]
        plt.plot(path[:, 0], path[:, 1])
    if goal is not None:
        for t in range(goal.shape[0]):
            goal_t = goal[t]
            plt.scatter(goal_t[0], goal_t[1])


    plt.show()
    plt.close()

def xy_to_paths(xy_paths):
    return [trajnetplusplustools.TrackRow(i, 0, xy_paths[i, 0].item(), xy_paths[i, 1].item(), 0, 0)
            for i in range(len(xy_paths))]

def viz(groundtruth, prediction, visualize, output_file=None):
    pred_paths = {}

    groundtruth = groundtruth.cpu().numpy().transpose(1, 0, 2)
    prediction = prediction.cpu().numpy().transpose(1, 0, 2)
    gt_paths = [xy_to_paths(path) for path in groundtruth]
    pred = [xy_to_paths(path) for path in prediction]

    pred_paths[0] = pred[0]
    pred_neigh_paths = None
    if visualize:
        pred_neigh_paths = {}
        pred_neigh_paths[0] = pred[1:]

    with show.predicted_paths(gt_paths, pred_paths, pred_neigh_paths, output_file):
        pass

def save_log(s, address):
    f = open(address, 'a+')
    f.write(s + '\n')
    f.close()
def erase_log(address):
  open(address, 'w').close()
def append_list_as_row(file_name, list_of_elem):  # appends a row to a csv file
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

def save_tensor_to_csv(filename, x):  # saves a tensor to a csv with name: filename in the main folder
    l = x.tolist()
    num_frames = len(l)
    num_agents = len(l[0])

    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    file1 = open(filename, 'w')
    s = '{"scene": {"id": 0, "p": 0, "s": 1, "e":' + str(num_frames) + ', "fps": 2.5, "tag": 0}}\n'
    file1.write(s)
    # print(filename)
    for agent in range(num_agents):
        for frame in range(num_frames):
            x = l[frame][agent][0]
            y = l[frame][agent][1]
            x = round(x, 2)
            y = round(y, 2)
            x = dont_nan(x)
            y = dont_nan(y)
            s = '{"track": {"f": ' + str(frame + 1) + ", " + '"p": ' + str(agent + 1) + ', "x": ' + str(
                x) + ', ' + '"y": ' + str(y) + '}}\n'
            file1.write(s)
            append_list_as_row(filename, [frame + 1, agent + 1, round(x, 2), round(y, 2)])

def calc_fde_ade(output, ground_truth):  # input: two tensors, returns fde, ade
    l = output.tolist()
    l2 = ground_truth.tolist()
    num_frames_output = len(l)
    num_frames_truth = len(l2)
    delta = num_frames_output - num_frames_truth
    distances = []
    for frame in range(num_frames_output):
        if frame + num_frames_truth >= num_frames_output:
            x1 = l[frame][0][0]  # for agent 0
            y1 = l[frame][0][1]
            x2 = l2[frame - delta][0][0]
            y2 = l2[frame - delta][0][1]
            d = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
            distances.append(d)
    # print(distances)
    return distances[-1], np.mean(distances)

def good_list(l):
    ans = []
    for i in l:
        if not math.isnan(i):
            ans.append(i)
    return ans
def seperate_xy(agent_path):
    xs = []
    ys = []
    for i in agent_path:
        xs.append(i[0])
        ys.append(i[1])
    return xs, ys

@contextmanager
def canvas(image_file, **kwargs):
    """Generic matplotlib context."""
    fig, ax = plt.subplots(**kwargs)
 
    yield ax

    fig.set_tight_layout(True)
    # type(image_file)
    
    if image_file:
        #print(image_file)
        image_file = image_file.replace(".png", ".pdf")
        #print(image_file)
        fig.savefig(image_file, dpi=250)
    
    fig.show()
    plt.close(fig)

def get_sizes(perturbed_path):
  len_limit = 21
  max_x = -10000
  min_x = 100000
  max_y = -100000
  min_y = 10000000
  for cnt, agent_path in enumerate(perturbed_path):
    xs, ys = seperate_xy(agent_path)
    #pdb.set_trace()
    if len(good_list(xs)) < len_limit:
        continue
    for j in range(0, len(xs)):
      max_x = max(max_x, xs[j])
      min_x = min(min_x, xs[j])
      max_y = max(max_y, ys[j])
      min_y = min(min_y, ys[j])
    lx = max_x - min_x
    ly = max_y - min_y
    
    if lx < ly:
      return 8.0 * lx / ly ,8.0
    else:
      return 8.0, 8.0 * ly / lx
def is_stationary(xs, ys):
    xs = good_list(xs)
    ys = good_list(ys)
    ans = 0
    for i in range(len(xs)):
      for j in range(i+1, len(xs)):
        dis = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
        ans = max(ans, dis)
    if ans > 4:
      return False
    return True
def get_char(x):
  if x < 10:
    return str(x)
  if x == 10:
    return 'A'
  if x == 11:
    return 'B'
  if x == 12:
    return 'C'
  if x == 13:
    return 'D'
  if x == 14:
    return 'E'
  if x == 15:
    return 'F'
def get_hex_s(x):
  return get_char(x // 16) + get_char(x % 16)
def get_hex(a, b, c):
  return '#' + get_hex_s(a) + get_hex_s(b) + get_hex_s(c)

@contextmanager
def paths(perturbed_path, real_path, output_file=None, collision_point_neighbor=None, collision_point_main=None):
    """Context to plot paths."""

    l1, l2 = 8, 8
    with canvas(output_file, figsize=(l1, l2)) as ax:
        #ax.grid(linestyle='dotted')
        #ax.set_aspect(1.0 , 'datalim')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        start_symbol = 'o'
        end_symbol = 's'

        yield ax
        obs_len = 9
        # other tracks

        len_limit = 21
        only_primary = False
        m_size_p = 3
        m_size_other = 4
        perturb_color = get_hex(235,51, 35)
        other_color = get_hex(119, 119, 119)
        #yellow_color = get_hex(111, 32, 110)
        yellow_color = 'magenta'
        for cnt, agent_path in enumerate(perturbed_path):
            
            xs, ys = seperate_xy(agent_path)
            #pdb.set_trace()
            if cnt > 0 and is_stationary(xs, ys):
              continue
            if len(good_list(xs)) < len_limit:
                continue
            if only_primary and cnt > 0:
                continue
            if cnt == 0:
                ax.plot(xs[0:1], ys[0:1], color=perturb_color , marker=start_symbol, linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color=perturb_color , marker=end_symbol, linestyle='None')

                ax.plot(xs[:obs_len], ys[:obs_len], color=perturb_color , linestyle='-')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color=perturb_color , linestyle='dotted')
                for j in range(1, obs_len):
                    ax.plot(xs[j], ys[j], color=perturb_color , marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)
                for j in range(obs_len - 1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color=perturb_color , marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)

            else:
                xs = good_list(xs)
                ys = good_list(ys)
                ax.plot(xs[0:1], ys[0:1], color=other_color, marker=start_symbol, linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color=other_color, marker=end_symbol, linestyle='None')

                
                ax.plot(xs[:obs_len], ys[:obs_len], color=other_color, linestyle='-')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color=other_color, linestyle='dotted')

                for j in range(1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color= other_color, marker='o', linestyle='None', zorder=0.9,
                            markersize=m_size_other)

        # real


        obs_len = 9
        # other tracks
        red_color = get_hex(143,147,83)
        for cnt, agent_path in enumerate(real_path):
            xs, ys = seperate_xy(agent_path)

            # markers
            if cnt == 0:
                ax.plot(xs[0:1], ys[0:1], color=red_color, marker=start_symbol, linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color=red_color, marker=end_symbol, linestyle='None')
                # track

                ax.plot(xs[:obs_len], ys[:obs_len], color=red_color, linestyle='-')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color=red_color, linestyle='dotted')

                for j in range(1, obs_len):
                    ax.plot(xs[j], ys[j], color=red_color, marker='o', linestyle='None', zorder=1.2, markersize=m_size_p)
                for j in range(obs_len - 1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color=red_color, marker='o', linestyle='None', zorder=1.2, markersize=m_size_p)
            else:
                continue
        orange_color = get_hex(210,97,42)
        if collision_point_neighbor != None and collision_point_main != None:
            x1 = collision_point_neighbor[0]
            y1 = collision_point_neighbor[1]
            x2 = collision_point_main[0]
            y2 = collision_point_main[1]
            dis = (x1 - x2)**2 + (y1 - y2)**2
            if dis < 0.09:
              ax.plot(collision_point_neighbor[0], collision_point_neighbor[1], color=yellow_color, marker='o', linestyle='None', zorder=0.9, markersize=m_size_p + 1.0)
              ax.plot(collision_point_main[0], collision_point_main[1], color=yellow_color, marker='o', linestyle='None', zorder=0.9, markersize=m_size_p + 1.0)


        # frame
        ax.set_facecolor(color = get_hex(233,242,245) )

@contextmanager
def paths_one(real_path, output_file=None):
    """Context to plot paths."""


    with canvas(output_file, figsize=(8, 8)) as ax:
        ax.grid(linestyle='dotted')
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        yield ax
        obs_len = 9
        # other tracks

        len_limit = 5
        only_primary = False
        m_size_p = 3
        m_size_other = 4

        for cnt, agent_path in enumerate(real_path):
            xs, ys = seperate_xy(agent_path)
            if len(good_list(xs)) < len_limit:
                continue
            if only_primary and cnt > 0:
                continue
            if cnt == 0:
                ax.plot(xs[0:1], ys[0:1], color='black', marker='x', linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color='black', marker='o', linestyle='None')
                # track
                ax.plot(xs[obs_len - 1:obs_len], ys[obs_len - 1:obs_len], color='green', marker='s', linestyle='None')

                ax.plot(xs[:obs_len], ys[:obs_len], color='black', linestyle='-')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color='black', linestyle='dotted')

                for j in range(1, obs_len):
                    ax.plot(xs[j], ys[j], color='green', marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)
                for j in range(obs_len - 1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color='blue', marker='o', linestyle='None', zorder=0.9, markersize=m_size_p)

            else:
                xs = good_list(xs)
                ys = good_list(ys)
                ax.plot(xs[0:1], ys[0:1], color='black', marker='x', linestyle='None')
                ax.plot(xs[-1:], ys[-1:], color='black', marker='o', linestyle='None')

                ax.plot(xs[:obs_len], ys[:obs_len], color='black', linestyle='-')
                ax.plot(xs[obs_len - 1:], ys[obs_len - 1:], color='black', linestyle='dotted')

                

                for j in range(1, len(xs) - 1):
                    ax.plot(xs[j], ys[j], color= '#FF00FFAA', marker='o', linestyle='None', zorder=0.9,
                            markersize=m_size_other)
        
        # frame
        tick_spacing = 1.0
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        #ax.legend()

