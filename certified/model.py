from baselines.model_d_pool import DPoolModel
from baselines.model_autobot import AutobotModel
from baselines.model_eq_motion import EqMotionModel
from utils import get_arguments
from dataset import Dataset



import matplotlib.pyplot as plt

import torch
import numpy as np

from tqdm import tqdm

class Model:
    
    def __init__(self,
                 model_type,
                 arguments,
                 device):
        
        self.arguments = arguments
        self.model_type = model_type
        self.device = device
        
        self.obs_length = self.arguments["general args"]["obs_length"]
        
        if self.model_type == 'd_pool':
            model_args = self.arguments["D-Pool args"]    
            self.predictor = DPoolModel(model_args, torch.device('cpu'))
        if self.model_type == 'autobot':
            model_args = self.arguments["Autobot args"]    
            self.predictor = AutobotModel(model_args, self.device)
        if self.model_type == 'eq_motion':
            model_args = self.arguments["EqMotion args"]
            self.predictor = EqMotionModel(model_args, self.device)
        
        
    
    def get_prediction(self, batch_scene, batch_goal):
        
        with torch.no_grad():
            pred = self.predictor.get_prediction(batch_scene, batch_goal)
        return pred
   
    
    def get_prediction_single_data(self, data):
        scene = data[1]
        goal = data[2]
        pred = self.predictor.get_prediction([scene], [goal])
        
        return pred[0, ...]