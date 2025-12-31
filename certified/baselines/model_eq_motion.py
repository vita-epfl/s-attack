# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:41:11 2023

"""

import torch
import numpy as np
from baselines.EqMotion.model_t import EqMotion


class EqMotionModel:
    
    def __init__(self, args, device):
        
        self.past_length = 9
        self.nf = 64
        self.channels = 64
        self.future_length = 12
        self.n_layers = 4
        self.norm_diff = False
        self.tanh = False
        
        self.device = device
        
        self.weights_path = args['weights_path']
                
        self.predictor = EqMotion(in_node_nf = self.past_length,
                                  in_edge_nf = 2,
                                  hidden_nf = self.nf,
                                  in_channel = self.past_length,
                                  hid_channel = self.channels,
                                  out_channel = self.future_length,
                                  device = self.device,
                                  n_layers = self.n_layers,
                                  recurrent = True,
                                  norm_diff = self.norm_diff,
                                  tanh = self.tanh)    
        
        model_ckpt = torch.load(self.weights_path)
        
        self.predictor.load_state_dict(model_ckpt['state_dict'], strict=False)
        self.predictor.eval()
        
        self.N = 3        
        self.constant = 1
        
        
    def get_prediction(self, batch_scene, batch_goal):
        
        batch_size = len(batch_scene)
        
        loc_batch = np.zeros((batch_size, self.past_length, self.N, 2))
        loc_end_batch = np.zeros((batch_size, self.future_length, self.N, 2))
        num_valid_batch = np.zeros((batch_size))
        
        for idx, (scene, goal) in enumerate(zip(batch_scene, batch_goal)):
        
        
            scene_eq = self.drop_ped_with_missing_frame(scene.numpy())
            curr_scene = self.drop_distant(scene_eq, max_num_peds = self.N)
            
            num_valid_scene = self.N
    
            if curr_scene.shape[1] < self.N:
                temp_curr_scene = np.zeros((21, self.N, 2))
                temp_curr_scene[:, :curr_scene.shape[1], :] = curr_scene
                curr_scene = temp_curr_scene.copy()
                num_valid_scene = curr_scene.shape[1]
            scene_eq = curr_scene
        
        
            loc_scene = scene_eq[:self.past_length]
            loc_end_scene = scene_eq[self.past_length:]
        
            loc_batch[idx] = loc_scene
            loc_end_batch[idx] = loc_end_scene
            num_valid_batch[idx] = num_valid_scene
        
        
        loc = torch.Tensor(loc_batch).to(self.device).permute(0,2,1,3)
        loc_end = torch.Tensor(loc_end_batch).to(self.device).permute(0,2,1,3)
        num_valid = torch.Tensor(num_valid_batch).to(dtype=torch.long, device = self.device)
        
        vel = torch.zeros_like(loc)
        vel[:,:,1:] = loc[:,:,1:] - loc[:,:,:-1]
        vel[:,:,0] = vel[:,:,1]
        
        vel = vel * self.constant
        nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
        loc_pred, category_list = self.predictor(nodes, loc.detach(), vel, num_valid)
        
        
        
        pred = torch.Tensor(loc_pred[:, 0, ...]).permute(0, 2, 1, 3).cpu()
        
        
        
        return pred
        
        
        

    def drop_ped_with_missing_frame(self, xy): # xy: (21, n, 68)
        """
        Drops pedestrians more than r meters away from primary ped
        """
        ## xy[scene_index, pedestrian_ID, 68]
        xy_n_t = np.transpose(xy, (1, 0, 2)) # (n, 21, 68)
        mask = np.ones(xy_n_t.shape[0], dtype=bool)
        for n in range(xy_n_t.shape[0]):
            # for t in range(xy_n_t.shape[1]):
            for t in range(9):
                if np.isnan(xy_n_t[n, t, 0]) == True:
                    mask[n] = False
                    break
        return np.transpose(xy_n_t[mask], (1, 0, 2))

    def drop_distant(self, xy, max_num_peds=5):
        """
        Only Keep the max_num_peds closest pedestrians
        """
        distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
        smallest_dist_to_ego = np.nanmin(distance_2, axis=0)
        
        return xy[:, np.argsort(smallest_dist_to_ego)[:(max_num_peds)]]


