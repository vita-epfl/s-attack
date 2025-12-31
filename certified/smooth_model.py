import torch
import numpy as np
from scipy.stats import norm
from scipy.signal import wiener

from dataset import Dataset
from model import Model
from utils import get_arguments, create_folders

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import pdb


class Denoiser:
    
    def __init__(self, denoiser_type, **kwargs):
        
        self.denoiser_type = denoiser_type
            
        if denoiser_type == "polynomial":
            if "polynomial_order" in kwargs:
                self.polynomial_order = kwargs["polynomial_order"]
            else:        
                self.polynomial_order = get_arguments()["polynomial denoiser args"]["polynomial_order"]
            
            
            
    
    def denoise(self, all_noisy_obs, sigma=0.1):
                            
        if self.denoiser_type == "wiener_filter":
            denoising_function = self.wiener_filter
        elif self.denoiser_type == "moving_average":
            denoising_function = self.moving_average
        elif self.denoiser_type == "polynomial":
            denoising_function = self.polynomial_denoiser
        
        
        res = torch.zeros_like(all_noisy_obs)
        for k in range(all_noisy_obs.shape[0]):
            
            denoised_obs = denoising_function(all_noisy_obs[k, ...])            
            res[k, ...] = denoised_obs
            
        
        return res
        
    
    
    
    @staticmethod
    def moving_average(obs):
        
        ego_tensor = obs[:, 0, :]
                
        ego = ego_tensor.clone().cpu().detach().numpy()
        
        window_size = 5
        
        smoothed_x = np.convolve(ego[:, 0],
                                 np.ones(window_size)/window_size,
                                 mode='valid')
        smoothed_y = np.convolve(ego[:, 1],
                                 np.ones(window_size)/window_size,
                                 mode='valid')
        
        denoised = np.column_stack((smoothed_x, smoothed_y))
            
        margin = int((window_size - 1) / 2)
        denoised_ego = np.copy(ego)
        denoised_ego[margin:-1 * margin, :] = denoised
        
        denoised_ego = torch.tensor(denoised_ego)
        
        denoised_obs = obs.clone()
        denoised_obs[:, 0, :] = denoised_ego
        
        return denoised_obs

    @staticmethod
    def wiener_filter(obs):
        
        ego_tensor = obs[:, 0, :]
        
        ego = ego_tensor.clone().cpu().detach().numpy()
        
        filtered_x = wiener(ego[:, 0])
        filtered_y = wiener(ego[:, 1])
        
        denoised = np.column_stack((filtered_x, filtered_y))
        denoised = denoised[1:-1, :]
        
        denoised_ego = np.copy(ego)
        denoised_ego[1:-1, :] = denoised
        
        denoised_ego = torch.tensor(denoised_ego)
        
        denoised_obs = obs.clone()
        denoised_obs[:, 0, :] = denoised_ego
                
        return denoised_obs
    
    
    def polynomial_denoiser(self, obs):
        
        order = self.polynomial_order
        
        ego_tensor = obs[:, 0, :]
        
        ego = ego_tensor.clone().cpu().detach().numpy()
        
        denoised_ego = np.zeros_like(ego)

        polynomial = np.poly1d(np.polyfit(ego[:,0], ego[:,1], deg=order))
        denoised_ego[:,0] = ego[:,0].copy()
        denoised_ego[:,1] = polynomial(ego[:,0])
        
        denoised_ego = torch.tensor(denoised_ego)
        
        denoised_obs = obs.clone()
        denoised_obs[:, 0, :] = denoised_ego
                
        return denoised_obs


class Evaluator:
    
    def __init__(self, evaluator_args):
        
        self.pred_length = evaluator_args["pred_length"]
        self.collision_threshold = evaluator_args["collision_threshold"]
        
        header_str = ""
        header_str += "ade\t"
        header_str += "fde\t"
        header_str += "collision\t"
        header_str += "cert_collision\t"
        header_str += "fbd\t"
        header_str += "abd\t"
        header_str += "\n"
        
        self.header_str = header_str
        
    
    def compute_metrics(self,
                        scene,
                        smoothed_pred,
                        certified_bounds):
        
        
        
        fde, ade = self.calc_fde_ade(smoothed_pred, scene[-self.pred_length:])
        fbd, abd = self.calc_fbd_abd(certified_bounds)
                
        cert_collision_2 = self.check_cert_collision(scene, certified_bounds)
                
        collision_2 = self.check_collision(smoothed_pred, scene[-self.pred_length:])
        
        metrics_str = ""
        metrics_str += "{}\t".format(ade)
        metrics_str += "{}\t".format(fde)
        metrics_str += "{}\t".format(collision_2)
        metrics_str += "{}\t".format(cert_collision_2)
        metrics_str += "{}\t".format(fbd)
        metrics_str += "{}\t".format(abd)
        
        metrics_str += "\n"

        return metrics_str
    
    
    def check_collision(self, pred, scene):
        dists = (pred - scene[:, 1:, :]).norm(dim = 2)
        coll = torch.abs(dists) < self.collision_threshold
        
        res = int(torch.sum(coll) > 0)
        
        return res
    
    def check_cert_collision(self, scene, certified_bounds):
        
        other_gt = scene[-self.pred_length:, 1:, :]
        
        lb_ego = certified_bounds[0, :, 0, :]
        ub_ego = certified_bounds[1, :, 0, :]
        
                
        for t in range(self.pred_length):
            x1_ego, y1_ego = lb_ego[t, :]
            x2_ego, y2_ego = ub_ego[t, :]
            ego = (x1_ego, y1_ego, x2_ego, y2_ego)
            for k in range(other_gt.shape[1]):
                x, y = other_gt[t, k, :]
                dist = self.find_distance_rectangle_point(ego, (x, y))
                
                if dist <= self.collision_threshold:
                    return 1
                
        return 0
    
    def find_distance_rectangle_point(self, rect, point):
        x_min, y_min, x_max, y_max = rect
        x, y = point
        
        dx = max(x_min - x, 0, x - x_max)
        dy = max(y_min - y, 0, y - y_max)
        
        dist = torch.Tensor([dx, dy]).norm()
        
        return dist
        
    def calc_fbd_abd(self, bounds:torch.Tensor):
        lb, ub = bounds[:, :, 0, :]
        
        box_dims = ub - lb
        
        diameters = box_dims.norm(dim = 1) / 2
        
        a_diameter = diameters.mean()
        
        f_diameter = diameters[-1]
            
        return f_diameter.item(), a_diameter.item()
    
    def calc_fde_ade(self, output, ground_truth):  # input: two tensors, returns fde, ade
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
        
        return distances[-1], np.mean(distances)


class MeanOperator:
    
    def __init__(self, r, sigma):        
        self.r = r
        self.sigma = sigma
    
    def aggregate(self, raw_data):
        res = torch.mean(raw_data, axis = 0)
        return res
    
    def compute_certified_bounds(self, smoothed_pred, raw_pred, init_boundaries):
        
        low_b, up_b = init_boundaries       
        agents_num = smoothed_pred.shape[1]
        low_b = low_b.repeat(agents_num, 1, 1).permute(1, 0, 2)
        up_b = up_b.repeat(agents_num, 1, 1).permute(1, 0, 2)
        
        small_sp = smoothed_pred.detach().cpu()
        small_low_b = low_b.cpu()
        small_up_b = up_b.cpu()        
                
        lb = small_low_b + (small_up_b-small_low_b)*norm.cdf(
            (self.eval_eta(small_sp, small_low_b, small_up_b) - self.r)/self.sigma
        )
    
        ub = small_low_b + (small_up_b-small_low_b)*norm.cdf(
            (self.eval_eta(small_sp, small_low_b, small_up_b) + self.r)/self.sigma
        )
        
        bounds = torch.stack((lb, ub), dim=0)
        
    
        return bounds
    
    def eval_eta(self, g, l, u):
        """
        evaluate the function named eta in the theory
        """
                                
        return self.sigma*norm.ppf(((g - l)/(u - l)).cpu())



class MedianOperator:
    
    def __init__(self, p, r, sigma):
        self.p = p
        self.r = r
        self.sigma = sigma
    
    
    def aggregate(self, raw_data):
        
        res = self.h_p(raw_data, self.p)
        return res
        
    
    def compute_certified_bounds(self, smoothed_pred, raw_pred, _):
        
        
                
        p_down = norm.cdf(norm.ppf(self.p) - self.r / self.sigma)
        p_up = norm.cdf(norm.ppf(self.p) + self.r / self.sigma)
        # p_down = 0.008
        # p_up = 1 - p_down
        
        lb = self.h_p(raw_pred, p_down)
        ub = self.h_p(raw_pred, p_up)
        
        lb = lb.cpu()
        ub = ub.cpu()
                
        bounds = torch.stack((lb, ub), dim=0)
        
        
        return bounds
    
    
    def h_p(self, x, p):
        y = torch.quantile(x, p, axis = 0)
        return y




class SmoothModel:
    
    def __init__(self, model, denoiser_type, r, sigma, smooth_model_args):
        
        
        self.model = model
        self.sigma = sigma
        self.r = r
                        
        self.n_monte_carlo = smooth_model_args["n_monte_carlo"]
        self.clamping_margin = smooth_model_args["clamping_margin"]
        self.obs_length = smooth_model_args["obs_length"]
        self.p_median = smooth_model_args["p"]
        self.max_r = smooth_model_args["max_r_for_clamping"]
                
        self.mean_operator = MeanOperator(self.r, self.sigma)
        self.median_operator = MedianOperator(self.p_median, self.r, self.sigma)
        
        self.init_bounds = self.compute_initial_bounds()
        
        self.denoiser_type = denoiser_type
        
        if denoiser_type is not None:
            self.denoiser = Denoiser(denoiser_type)            
        else:
            self.denoiser = None
        
    
    
    def compute_initial_bounds(self):
        
        min_y = [-1.479 , -1.72999954, -2.5       , -3.350, -4.05,  -4.88, -5.6799, -6.42983, -7.1300, -7.8299,   -8.6800, -9.829]
        max_y = [1.13000011, 1.84999943, 2.77999973, 3.57999897, 4.40000057,
               5.30000019, 6.15000057, 7.17999935, 7.82999897, 8.47000027,
               9.41999912, 9.91999912] 
        min_x  = [-1.04999971, -1.76999998, -2.53999996, -3.17000008, -3.95000005,
               -4.69000006, -5.47000027, -6.07000017, -6.92000008, -7.5999999 ,
               -8.38000011, -9.07999992]
        max_x = [ 1.21999931,  2.19000053,  2.85000038,  3.93999958,  5.09999943,
                6.31999969,  7.59999943,  8.93999958, 10.32999992, 11.78999996,
               13.28999996, 14.84999943]
        
                
        low = torch.tensor([min_x, min_y]).transpose(1, 0)
        high = torch.tensor([max_x, max_y]).transpose(1, 0)
        
        low = low - self.max_r
        high = high + self.max_r
        
        diff = high - low
        
        low = low - diff * self.clamping_margin / 2
        high = high + diff * self.clamping_margin / 2
        
        return (low, high)
        
                
    
    
    def get_smoothed_results(self, data, return_details = False):        
        
        
        scene = data[1].clone()
        goal = data[2].clone()
        
        noisy_scenes = self.produce_all_noisy_scenes(scene)
        
        noisy_obs = noisy_scenes[:, :self.obs_length, ...]
        if self.denoiser is not None:
            denoised_abs = self.denoiser.denoise(noisy_obs, self.sigma)
        else:
            denoised_abs = noisy_obs
        
        denoised_scenes = noisy_scenes.clone()
        denoised_scenes[:, 0:self.obs_length, ...] = denoised_abs
        
        outputs_perturbed = self.get_raw_results(denoised_scenes, goal)
        
        smoothed_pred_all = {}
        certified_bounds_all = {}
        
        for operator_name in ["mean", "median"]:
            
            if operator_name == "mean":
                
                operator = self.mean_operator
                outputs_perturbed_clamped, init_bounds = self.clamp(outputs_perturbed, scene)
                smoothed_pred = operator.aggregate(outputs_perturbed_clamped)
                certified_bounds = operator.compute_certified_bounds(smoothed_pred,
                                                                     outputs_perturbed_clamped,
                                                                     init_bounds)
                
                
                
            elif operator_name == "median":
                operator = self.median_operator
                smoothed_pred = operator.aggregate(outputs_perturbed)
                certified_bounds = operator.compute_certified_bounds(smoothed_pred,
                                                                     outputs_perturbed,
                                                                     self.init_bounds)
            
            smoothed_pred_all[operator_name] = smoothed_pred
            certified_bounds_all[operator_name] = certified_bounds
        
        if return_details:
            
            outputs_perturbed_dict = {"mean" : outputs_perturbed_clamped,
                                      "median" : outputs_perturbed}
            
            res = (smoothed_pred_all,
                   certified_bounds_all,
                   self.init_bounds,
                   noisy_scenes,
                   outputs_perturbed_dict)
            return res
        else:
            return smoothed_pred_all, certified_bounds_all
    
    def get_raw_results(self, all_scenes, goal):
        
        batch_scene = []
        batch_goal = []
        for k in range(all_scenes.shape[0]):
            batch_scene.append(all_scenes[k, ...])
            batch_goal.append(goal)
        
        all_outputs = self.model.get_prediction(batch_scene, batch_goal)
        
        return all_outputs
        
    
    def produce_all_noisy_scenes(self, scene):
        
        all_noisy_scenes = []
        
        for k in range(self.n_monte_carlo):
        
            noise = torch.zeros_like(scene)
            noisy_part = noise[:self.obs_length, 0, :].normal_(mean = 0, std = self.sigma)        
            noise[:self.obs_length, 0, :] = noisy_part
            
            if self.sigma == 0.0:
                noise = torch.zeros_like(scene)
            
            noisy_scene = scene + noise
            
            all_noisy_scenes.append(noisy_scene.clone())
        
        all_noisy_scenes = torch.stack(all_noisy_scenes)
        
        
        return all_noisy_scenes
    
    
    
        
    
        
    def clamp(self, raw_pred, scene):
        
        low, high = self.init_bounds
        
        normalization_transition = scene[self.obs_length - 1, 0, :]
        
        repeated_low = low.repeat(raw_pred.shape[0], 1, 1)
        repeated_high = high.repeat(raw_pred.shape[0], 1, 1)
        
        normalized_raw_pred = raw_pred.clone() - normalization_transition
        
        clamped_pred = normalized_raw_pred
        clamped_pred[:, :, 0, :] = torch.maximum(repeated_low, clamped_pred[:, :, 0, :])
        clamped_pred[:, :, 0, :] = torch.minimum(repeated_high, clamped_pred[:, :, 0, :])
        
        clamped_pred = normalized_raw_pred + normalization_transition
        
        bounds = (low + normalization_transition,
                  high + normalization_transition)
                
        return clamped_pred, bounds

