from trajnetplusplustools.trajnetplusplustools import Reader

import torch
import numpy as np
import pickle
import os
import math
from operator import itemgetter
from utils import get_arguments


def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask


def prepare_data(path, subset='/train/', sample=1.0, goals=True):
    """ Prepares the train/val scenes and corresponding goals """

    ## read goal files
    all_goals = {}
    all_scenes = []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
    ## Iterate over file names
        
    for file in files:
        reader = Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
        if goals:
            goal_dict = pickle.load(open('dest_new/' + subset + file + '.pkl', "rb"))
            ## Get goals corresponding to train scene
            all_goals[file] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scene}
        all_scenes += scene

    if goals:
        return all_scenes, all_goals
    return all_scenes, None


def seperate_xy(agent_path):
    xs = []
    ys = []
    for i in agent_path:
        xs.append(i[0])
        ys.append(i[1])
    return xs, ys


def good_list(l):
    ans = []
    for i in l:
        if not math.isnan(i):
            ans.append(i)
    return ans


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


class Dataset:
    
    def __init__(self, dataset_args):
        
        path = dataset_args["path"]
        data_part = dataset_args["data_part"]        
        sample = dataset_args["sample"]
        goals = dataset_args["goals"]
        remove_static = dataset_args["remove_static"] if "remove_static" in dataset_args else False
        
        if data_part == 'test':
            test_scenes, test_goals = prepare_data(path,
                                                   subset = '/test/',
                                                   sample = sample,
                                                   goals = goals)
        elif data_part == 'train':
            test_scenes, test_goals = prepare_data(path,
                                                   subset = '/train/',
                                                   sample = sample,
                                                   goals = goals)
        elif data_part == 'secret': #NEW - UNTRACKED - ONLY HERE LOCALLY
            test_scenes, test_goals = prepare_data(path,
                                                   subset = '/test_private/',
                                                   sample = sample,
                                                   goals = goals)
        
        
        
        self.test_scenes = test_scenes
        self.test_goals = test_goals
        self.device = torch.device("cpu")
        self.label = data_part
        self.remove_static = remove_static
        self.all_data = self.load_preprocessed_scenes()

    def load_preprocessed_scenes(self):
        
        
        scenes = self.test_scenes
        goals = self.test_goals
        
        all_data = []
        for i, (filename, scene_id, paths) in enumerate(scenes):
            scene = Reader.paths_to_xy(paths) # Now T_obs x N_agent x 2
            if goals is not None:
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])
            
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]
            
            scene = torch.Tensor(scene).to(self.device)
            scene_goal = torch.Tensor(scene_goal).to(self.device)
            
            ## remove stationnary
            if self.remove_static:
                valid_scene = True
                for agent_path in paths:
                    xs, ys = seperate_xy(agent_path)
                    if is_stationary(xs, ys): #one or more is stationary
                        valid_scene = False #we skip this scnene
                #print(valid_scene)
                if not valid_scene:
                    continue
            
            all_data.append((scene_id, scene, scene_goal))
        
        all_data = sorted(all_data, key = itemgetter(0))
        
        return all_data
    
    def __getitem__(self, idx):        
        return self.all_data[idx]
    
    def __len__(self):
        return len(self.all_data)
