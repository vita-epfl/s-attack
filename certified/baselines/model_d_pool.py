# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:05:53 2023

"""

import torch
from baselines.DPool.non_gridbased_pooling import HiddenStateMLPPooling, NN_LSTM, SAttention
from baselines.DPool.lstm import LSTM

class DPoolModel:
    
    def __init__(self, model_args, device):
        
        
        self.device = device
        self.pred_length = model_args["pred_length"]
        self.obs_length = model_args["obs_length"]
        
        self.coordinate_embedding_dim = model_args["coordinate_embedding_dim"]
        self.hidden_dim = model_args["hidden_dim"]
        self.goals = model_args["goals"]
        self.goal_dim = model_args["goal_dim"]
        self.weights_path = model_args["weights_path"]
        
        self.model_type = 'd_pool'
        self.pool_dim = model_args["pool_dim"]
        self.vel_dim = model_args["vel_dim"]
        self.neigh = model_args["neigh"]
        
        
        self.model = self.load_model()
        
    def get_prediction_single_scene(self, scene, scene_goal):
        
        observed = scene[:self.obs_length].detach().clone().to(self.device)
        batch_split = torch.Tensor([0,scene.size(1)]).to(self.device).long()
                
                                
        _, pred = self.model(observed.clone(),  
                             scene_goal, 
                             batch_split, 
                             n_predict = self.pred_length)
        
        
        
                
        pred_ego = pred[-self.pred_length:, 0:1, :]
                
        return pred_ego
    
    def get_prediction(self, batch_scene, batch_goal):
        
        res = []
        
        for scene, goal in zip(batch_scene, batch_goal):
            pred = self.get_prediction_single_scene(scene, goal)
            res.append(pred)
        
        res = torch.stack(res)
        return res
    
    def load_model(self):
        
        
        pool = self.load_pool()
        
        model = LSTM(pool = pool,
                     embedding_dim = self.coordinate_embedding_dim,
                     hidden_dim = self.hidden_dim,
                     goal_flag = self.goals,
                     goal_dim = self.goal_dim)
        
        
        load_address = self.weights_path
        # print("Model load address : ", load_address)
        
        with open(load_address, 'rb') as f:
            checkpoint = torch.load(f)
        pretrained_state_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_state_dict, strict=False)
        
        
        model.to(self.device)
        
        for p in model.parameters():
            p.requires_grad = False
        
        model.eval()
        
        return model
    
    
    def load_pool(self):
        
        pool = None
        if self.model_type == 'hiddenstatemlp':
            pool = HiddenStateMLPPooling(hidden_dim = self.hidden_dim,
                                         out_dim = self.pool_dim,
                                         mlp_dim_vel = self.vel_dim)
        elif self.model_type == 'd_pool':  # always this one
            pool = NN_LSTM(n = self.neigh,
                           hidden_dim = self.hidden_dim,
                           out_dim = self.pool_dim)
        elif self.model_type == 's_att':
            pool = SAttention(hidden_dim = self.hidden_dim,
                              out_dim = self.pool_dim)
        
        return pool

