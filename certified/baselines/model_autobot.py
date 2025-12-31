# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:42:33 2023

"""

import torch
from baselines.Autobot.models.autobot_joint import AutoBotJoint
from baselines.Autobot.lstm_codebase import LSTMPredictor

from joblib import Parallel, delayed


class AutobotModel:
    
    def __init__(self, args, device):
        
        
        self.k_attr = 2
        self.hidden_size = 128
        self.num_other_agents = 5
        self.num_modes = 1
        self.pred_horizon = 12
        self.num_encoder_layers = 2
        self.dropout = 0.1
        self.tx_num_heads = 16
        self.num_decoder_layers = 2
        self.tx_hidden_size = 384
        self.map_attr = 0
        self.num_agent_types = 1
        self.predict_yaw = False
        
        self.device = device
        
        self.pred_length = args['pred_length']
        self.obs_length = args['obs_length']
        self.modes = args['modes']
        self.weights_path = args['weights_path']
        
        self.jobs_number = args["jobs number"]
        
        
        self.autobot_model = AutoBotJoint(k_attr=self.k_attr,
                                          d_k=self.hidden_size,
                                          _M=self.num_other_agents,
                                          c=self.num_modes,
                                          T=self.pred_horizon,
                                          L_enc=self.num_encoder_layers,
                                          dropout=self.dropout,
                                          num_heads=self.tx_num_heads,
                                          L_dec=self.num_decoder_layers,
                                          tx_hidden_size=self.tx_hidden_size,
                                          use_map_lanes=False,
                                          map_attr=self.map_attr,
                                          num_agent_types=self.num_agent_types,
                                          predict_yaw=self.predict_yaw).to(self.device)
        
        
        state_dict = torch.load(self.weights_path)['state_dict']
        self.autobot_model.load_state_dict(state_dict)
        
        self.predictor = LSTMPredictor(self.autobot_model)
        self.predictor.model.to(device)
        
    def get_prediction_single_scene(self, scene, scene_goal):
        
        
        model_prediction = self.predictor(scene,
                                          scene_goal,
                                          n_predict = self.pred_length,
                                          obs_length = self.obs_length,
                                          modes = self.modes,
                                          args = None)
        
        
        new_size = (model_prediction.shape[0],
                    scene.shape[1],
                    model_prediction.shape[2],)
        modified_size = torch.zeros(new_size) + float('nan')
        
        if scene.shape[1] > model_prediction.shape[1]:            
            modified_size[:, :model_prediction.shape[1], :] = model_prediction
        else:
            modified_size = model_prediction[:, :scene.shape[1], :]
        
        
        res = modified_size[:, 0:1, :]
        
        return res
    
    
    def get_prediction(self, batch_scene, batch_goal):
        
        pred_list = Parallel(n_jobs = self.jobs_number)(delayed(self.get_prediction_single_scene)(scene, goal) for scene, goal in zip(batch_scene, batch_goal))
        
        res = torch.stack(pred_list)
        
        return res
    




