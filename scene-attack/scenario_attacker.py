import torch
import numpy as np
import copy
import glob
from scipy.interpolate import CubicSpline
from scipy.ndimage.morphology import distance_transform_edt
from shapely.geometry import LineString
from shapely.affinity import affine_transform, rotate
from argparse import Namespace

from attackedmodels.UberLaneGCN import lanegcn
from attackedmodels.UberLaneGCN import data_for_attack
from attackedmodels.UberLaneGCN.utils import load_pretrain, gpu
from attackedmodels.UberLaneGCN.preprocess_data import to_long, preprocess
from attackedmodels.UberLaneGCN.data import ref_copy, from_numpy, collate_fn
from attackedmodels.DATF import datf_argo
from attackedmodels.WIMP import wimp_argo
from attackedmodels.MPC import MPC_model
from attackedmodels.WIMP.src.models.WIMP import WIMP as wimpmodel
from attack_functions import Combination
from config import *
from visualization_utils import viz_scenario
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.se2 import SE2



class SceneAugmentor:
    def __init__(self, argo_path=get_argo_val_path(), ckpt_path=get_lanegcn_path(), correct_speed=True, render=False, models_name="LaneGCN"):
        super(SceneAugmentor, self).__init__()
        self.models_name = models_name
        self.am = ArgoverseMap()

        if models_name == "LaneGCN":
            _, _, _, net, _, _, _ = lanegcn.get_model()

            # load pretrained model
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            load_pretrain(net, ckpt["state_dict"])
            net.eval()
            self.attacking_net = net
        elif models_name == "DATF":
            self.maps = {}
            self.maps['MIA'] = {}
            self.maps['PIT'] = {}
            for city_name in ['MIA', 'PIT']:
                map_mask, image_to_city = self.am.get_rasterized_driveable_area(city_name)
                image = map_mask.astype(np.int32)
                invert_image = 1-image
                dt = np.where(invert_image, -distance_transform_edt(invert_image), distance_transform_edt(image))
                self.maps[city_name]['dt'] = dt
                self.maps[city_name]['mask'] = map_mask
                self.maps[city_name]['image_to_city'] = image_to_city
                self.maps[city_name]['drivable_area'] = map_mask
                

            # load pretrained model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.DATF_obj = datf_argo.DATF_argoverse_dataset(self.maps)
            model = self.DATF_obj.get_model()
            path = get_datf_path()
            names = np.sort(np.array(glob.glob(path + "*/*.pth.tar")))
            ckpt = names[-1]
            print(ckpt)
            checkpoint = torch.load(ckpt)
            model.load_state_dict(checkpoint['model_state'], strict=True)
            model = model.to(device)
            self.attacking_net = model
        elif models_name == "WIMP":
            # load pretrained model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.WIMP_obj = wimp_argo.WIMP_argoverse_dataset(argo_path = argo_path)
            path = get_wimp_path()
            names = np.sort(np.array(glob.glob(path + '*')))
            ckpt = names[-1]
            print(ckpt)
            argumants = Namespace(IFC=True, add_centerline=False, attention_heads=4, batch_norm=False, batch_size=25, check_val_every_n_epoch=3,
                         dataroot='/scratch/izar/mshahver/wimp_pre2', dataset='argoverse', distributed_backend='ddp', dropout=0.5,
                         early_stop_threshold=5, experiment_name='example', gpus=2, gradient_clipping=True, graph_iter=1, hidden_dim=512,
                         hidden_key_generator=True, hidden_transform=False, input_dim=2, k_value_threshold=10, k_values=[6, 5, 4, 3, 2, 1],
                         lr=0.0001, map_features=False, max_epochs=120, mode='val', model_name='WIMP', no_heuristic=False, non_linearity='relu',
                         num_layers=4, num_mixtures=6, num_nodes=1, output_conv=True, output_dim=2, output_prediction=True, precision=32,
                         predict_delta=False, resume_from_checkpoint=None, scheduler_step_size=[60, 90, 120, 150, 180], seed=None, segment_CL=False,
                         segment_CL_Encoder=False, segment_CL_Encoder_Gaussian=False, segment_CL_Encoder_Gaussian_Prob=False, segment_CL_Encoder_Prob=True,
                         segment_CL_Gaussian_Prob=False, segment_CL_Prob=False, use_centerline_features=True, use_oracle=False, waypoint_step=5,
                         weight_decay=0.0, workers=8, wta=False)
            
            model = wimpmodel(argumants)
            model = model.load_from_checkpoint(ckpt, strict=True)
            model = model.to(device)
            self.attacking_net = model
        elif models_name == "MPC":
            pass
        else:
            raise Exception("The specified model's name is not in the options list. Please choose a correct model name.")
        self.attack_params = None
        self.attack_function = None
        self.scale_factor = 1

        self.scenario_idx = -1  # the current scenario
        self.ARGO_VAL_SIZE = 39_472

        argo_path=get_argo_val_path()
        self.avl = ArgoverseForecastingLoader(argo_path)
        self.avl.seq_list = sorted(self.avl.seq_list)

        self.agent_pred = None
        self.correct_speed = correct_speed
        self.visualize = render
        
    def get_scenario(self):
        # make some data of trajectories
        self.data = data_for_attack.read_argo_data(self.scenario_idx, self.avl)
        self.data = data_for_attack.get_obj_feats(self.data)
        
        if self.models_name == "LaneGCN":
            self.get_lanegcn_agent_pred()
        elif self.models_name == "DATF":
            self.get_DATF_agent_pred()
        elif self.models_name == "WIMP":
            self.get_WIMP_agent_pred()
        elif self.models_name == "MPC":
            self.get_MPC_agent_pred()
        else:
            raise Exception("Wrong model name.")

    def get_MPC_agent_pred(self):
        for gt_pred in self.data['gt_preds']:
            # first rotate ground truth like it is done in data for the prediction
            gt_pred[:, :] = np.matmul(self.data['rot'], (gt_pred - self.data['orig'].reshape(-1, 2)).T).T
            # skew the ground truths
            gt_pred[:, :] = self.apply_transform_function(gt_pred)
            # undo the rotation to bring the ground truth back to the world coordinates
            gt_pred[:, :] = np.matmul(self.data['rot'].T, gt_pred.T).T + self.data['orig'].reshape(-1, 2)

        # next, make the data of scene
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = [-100.0, 100.0, -100.0, 100.0]
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(self.data['orig'][0], self.data['orig'][1], self.data['city'], radius)
        lane_ids = copy.deepcopy(lane_ids)

        self.lane_centerlines = []
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[self.data['city']][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(self.data['rot'], (lane.centerline - self.data['orig'].reshape(-1, 2)).T).T
            centerline = self.apply_transform_function(centerline)
            self.lane_centerlines.append(centerline)

        obs = self.data["trajs"][0][:20]
        candidate_paths = self.am.get_candidate_centerlines_for_traj(obs, self.data["city"], viz=False)
        best_path = candidate_paths[self.find_best_lane(candidate_paths, obs[-1])]
        # process obs and best_path
        obs = np.matmul(self.data['rot'], (obs - self.data['orig'].reshape(-1, 2)).T).T
        if self.correct_speed:
            obs = self.correct_history(obs)
        best_path = np.matmul(self.data['rot'], (best_path - self.data['orig'].reshape(-1, 2)).T).T
        best_path = self.apply_transform_function(best_path)
        best_path = best_path[best_path[:, 0] >= 0, :]

        self.agent_pred = MPC_model.get_pred(best_path, obs)
        self.agent_pred = np.matmul(self.data['rot'].T, self.agent_pred.T).T + self.data['orig'].reshape(-1, 2)

    def find_best_lane(self, lanes, orig=np.array([0, 0])):
        best, min_score = 0, 10000
        for i in range(len(lanes)):
            if self.is_straight(lanes[i]):
                min_dist = 10000
                for point in lanes[i]:
                    point_trans = point - orig
                    min_dist = min(min_dist, np.sqrt((point_trans ** 2).sum()))
                lane_length = 0
                for t in range(len(lanes[i]) - 1):
                    lane_length += self.dist(lanes[i][t], lanes[i][t + 1])
                score = min_dist + 8 * lane_length / self.dist(lanes[i][0], lanes[i][-1])
                if score < min_score:
                    best = i
                    min_score = score
        return best

    def dist(self, point1, point2):
        return np.sqrt(((point1 - point2) ** 2).sum())

    def is_straight(self, lane_points):
        eps = 5
        x, y = lane_points[:, 0], lane_points[:, 1]
        w, b = np.polyfit(x, y, deg=1)
        if ((y - w * x - b) ** 2).sum() <= eps:
            return True
        else:
            return False

    def get_lanegcn_agent_pred(self):
        """
        1 - Gets the current scenario in dataset
        2 - Applies scene change (changing map and vehicle trajectory history and gt)
        3 - Find the prediction of model on the changed scene.
        """

        idx = self.scenario_idx

        # first make the data of trajectories
        data = data_for_attack.read_argo_data(idx, self.avl)
        data = data_for_attack.get_obj_feats(data)
        step0 = data['step0']
        for i, feat in enumerate(data["feats"]):
            feat[:, :2] = self.apply_transform_function(feat[:, :2])
            if i == 0 and self.correct_speed:  # apply speed correction on the history of agent if the correct_speed flag is on
                feat[:, :2] = self.correct_history(feat[:, :2])
            feat[1:, :2] -= feat[:-1, :2]
            feat[step0[i], :2] = 0

        data['ctrs'] = self.apply_transform_function(data['ctrs'])

        for i, gt_pred in enumerate(data['gt_preds']):
            # first rotate ground truth like it is done in data for the prediction
            gt_pred[:, :] = np.matmul(data['rot'], (gt_pred - data['orig'].reshape(-1, 2)).T).T
            # skew the ground truths
            gt_pred[:, :] = self.apply_transform_function(gt_pred)
            if i == 0:  # applies gt correction given the scale_factor calculated in history correction. Note that if
                # the speed correction is off, scale_factor is equal to 1 so this won't change the gt
                gt_pred[:, :] = self.correct_gt(gt_pred, self.scale_factor)
            # undo the rotation to bring the ground truth back to the world coordinates
            gt_pred[:, :] = np.matmul(data['rot'].T, gt_pred.T).T + data['orig'].reshape(-1, 2)

        # next, make the data of scene
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = [-100.0, 100.0, -100.0, 100.0]
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
        lane_ids = copy.deepcopy(lane_ids)
        lanes = dict()
        self.lane_centerlines = []
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            centerline = self.apply_transform_function(centerline)
            self.lane_centerlines.append(centerline)
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lane.polygon = self.apply_transform_function(lane.polygon[:, :2])
                lanes[lane_id] = lane

        data['graph'] = data_for_attack.get_lane_graph(data, self.am, lanes)
        data['idx'] = idx

        # now the real pre processing begins (only pre processing the graph)
        graph = dict()
        for key in ['lane_idcs', 'ctrs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'feats']:
            graph[key] = ref_copy(data['graph'][key])
        graph['idx'] = idx
        out = preprocess(to_long(gpu(dict(from_numpy(graph)))), 6)
        data['graph']['left'] = out['left']
        data['graph']['right'] = out['right']
        data = dict(collate_fn([copy.deepcopy(data)]))
        with torch.no_grad():
            output = self.attacking_net(data)
        self.agent_pred = output["reg"][0][0:1].detach().cpu().numpy().squeeze()[0]

    def correct_history(self, history):  # inputs history points in the agents coordinate system and corrects its speed   
        # calculating the minimum r of the attacking turn
        border1 = self.attack_params["smooth-turn"]["border"]
        border2 = self.attack_params["double-turn"]["border"]
        border3 = self.attack_params["ripple-road"]["border"]
        l_range = min(border1, border2, border3)
        r_range = max(border1 + 10, border2 + 20 + self.attack_params["double-turn"]["l"],
                      border3 + self.attack_params["ripple-road"]["l"])

        search_points = np.linspace(l_range, r_range, 100)
        search_point_rs = self.calc_radius(search_points)
        min_r = search_point_rs.min()
        g = 9.8
        miu_s = 0.7
        max_speed = np.sqrt(miu_s * g * min_r)
        self.current_speed = np.sqrt(((history[-1] - history[-2]) ** 2).sum()) * 10
        if self.current_speed <= max_speed:
            return history
        self.scale_factor = max_speed / self.current_speed
        return history * self.scale_factor

    def get_DATF_agent_pred(self):
        """
        1 - Gets the current scenario in dataset
        2 - Applies scene change (changing map and vehicle trajectory history and gt)
        3 - Find the predictions of DATF model on the changed scene.
        """
        # Don't run the map transform function when the attack powers are equal to zero
        sum_power = np.abs(self.attack_params["smooth-turn"]["attack_power"]) + np.abs(self.attack_params["double-turn"]["attack_power"]) + np.abs(self.attack_params["ripple-road"]["attack_power"])
        baseline = (sum_power == 0)
        if(not baseline):
            self.map_transform()
            
        # Load and preprocess it data
        df = copy.deepcopy(self.avl[self.scenario_idx].seq_df)
        obsv_df, pred_df = self.DATF_obj.generate_trajectories(df)

        # here we change the trajectories
        ag = obsv_df[obsv_df.OBJECT_TYPE == 'AGENT']
        ag_points = ag[['X', 'Y']].to_numpy()
        ag_points = self.correct_history(ag_points)
        obsv_df[obsv_df.OBJECT_TYPE == 'AGENT'][['X', 'Y']] = ag_points
        if(not baseline):
            obsv_df = self.points_transform(obsv_df)
            pred_df = self.points_transform(pred_df)
        agent_history = copy.deepcopy(np.array(obsv_df[obsv_df.OBJECT_TYPE == "AGENT"][['X', 'Y']]))
        
        obsv_traj, obsv_traj_len, pred_traj, pred_traj_len, decoding_agents_mask, decode_start_pos, decode_start_vel, metadata = self.DATF_obj.prepare_samples(self.scenario_idx, obsv_df, pred_df)
        context_map, prior_map, vis_map = self.DATF_obj.getmaps(metadata, self.maps)
        
        self.context_map = context_map
        self.prior_map = prior_map
        self.vis_map = vis_map
        
        model_input = (obsv_traj, obsv_traj_len, pred_traj, pred_traj_len, decoding_agents_mask, decode_start_pos, decode_start_vel, context_map, prior_map, vis_map, metadata)
        model_input = self.DATF_obj.collate(model_input)
        
        agent_obsv_num = 0
        obsv = model_input[0]
        
        for i, traj in enumerate(obsv):
            temp = np.around(traj.numpy().reshape(1, -1)[0].astype(np.float), decimals = 2)
            mask = np.array([temp[j] in np.around(agent_history, decimals = 2) for j in range(temp.shape[0])])
            if (np.ndarray.all(mask)):
                agent_obsv_num = i
        agent_pred_num = np.count_nonzero(np.array(model_input[6])[:agent_obsv_num + 1] == True)
        
        with torch.no_grad():
            pred_points = self.DATF_obj.run_model(self.attacking_net, model_input).cpu().numpy()
        self.agent_preds = pred_points[agent_pred_num - 1]
        self.agent_preds += np.array(self.data['orig'])
        self.agent_gt = np.array(self.data['gt_preds'][0])[[4, 9, 14, 19, 24, 29]]
        
        self.all_preds = self.agent_preds.copy()

    def points_transform(self, obsv_df):
        points = obsv_df[['X', 'Y']].to_numpy()
        points = np.matmul(self.data['rot'], points.reshape(-1, 2).T).T
        points = self.apply_transform_function(points)
        points = np.matmul(self.data['rot'].T, points.reshape(-1,2).T).T
        obsv_df[['X', 'Y']] = points
        return obsv_df
    
    def map_transform(self):
        """
        change the self.map object by transform function.

        Returns
        -------
        None.

        """
        # drivable points in image
        dr_area = np.array(copy.deepcopy(self.maps[self.data['city']]['drivable_area']))
        dr_points = np.argwhere(dr_area > 0)
        # turn them into city coordinate
        image_coords = np.round(dr_points[:, :2]).astype(np.int64)
        
        image_coords[:, [0, 1]] = image_coords[:, [1, 0]]

        se2_rotation = self.maps[self.data['city']]['image_to_city'][:2, :2]
        se2_trans = self.maps[self.data['city']]['image_to_city'][:2, 2]

        npyimage_to_city_se2 = SE2(rotation=se2_rotation, translation=se2_trans)

        agent_image_his = npyimage_to_city_se2.transform_point_cloud(self.data['trajs'][0][:20, ])

        image_coords = image_coords[image_coords[:, 0] > agent_image_his[19][0] - 400]
        image_coords = image_coords[image_coords[:, 0] < agent_image_his[19][0] + 400]
        image_coords = image_coords[image_coords[:, 1] > agent_image_his[19][1] - 400]
        image_coords = image_coords[image_coords[:, 1] < agent_image_his[19][1] + 400]

        city_coords = npyimage_to_city_se2.inverse_transform_point_cloud(image_coords)
        city_coords = np.matmul(self.data['rot'], (city_coords - self.data['orig'].reshape(-1, 2)).T).T
        city_coords = self.apply_transform_function(city_coords)
        city_coords = np.matmul(self.data['rot'].T, city_coords.T).T + self.data['orig'].reshape(-1, 2)
        image_coords = npyimage_to_city_se2.transform_point_cloud(city_coords)
        
        image_coords[:, [0, 1]] = image_coords[:, [1, 0]]

        image_coords = image_coords.astype(int)
        image_coords = image_coords[image_coords[:,0] > 0]
        image_coords = image_coords[image_coords[:,1] > 0]
        image_coords = image_coords[image_coords[:,0] < dr_area.shape[0]]
        image_coords = image_coords[image_coords[:,1] < dr_area.shape[1]]
        map_mask = np.zeros_like(dr_area)
        map_mask[image_coords[:,0], image_coords[:,1]] = 1
        self.maps[self.data['city']]['mask'] = map_mask
        image = map_mask.astype(np.int32)
        invert_image = 1-image
        dt = np.where(invert_image, -distance_transform_edt(invert_image), distance_transform_edt(image))
        self.maps[self.data['city']]['dt'] = dt
        self.maps[self.data['city']]['mask'] = map_mask
    
    def get_WIMP_agent_pred(self):
        """
        1 - Gets the current scenario in dataset
        2 - Applies scene change (changing map and vehicle trajectory history and gt)
        3 - Find the predictions of WIMP model on the changed scene.
        """
        model_input = self.WIMP_obj.preprocess(self.scenario_idx, self.data['orig'], self.data['rot'], self.attack_params, self.avl.seq_list[self.scenario_idx])
        with torch.no_grad():
            predictions, out2, out3 = self.attacking_net(**model_input)
        predictions = predictions.cpu().numpy()
        predictions = predictions[0][0]    
        xy_locations = self.data['trajs'][0][:20, :]
        xy_locations = np.matmul(self.data['rot'], (xy_locations - self.data['orig'].reshape(-1, 2)).T).T
        xy_locations = self.correct_history(xy_locations)
        xy_locations = np.matmul(self.data['rot'].T, xy_locations.T).T + self.data['orig'].reshape(-1, 2)
        trajectory = LineString(xy_locations)
        translation = np.array([xy_locations[0, 0], xy_locations[0, 1]])
        rotation = np.degrees(np.arctan2(xy_locations[-1][1] - self.data['trajs'][0][0][1],
                                   xy_locations[-1][0] - self.data['trajs'][0][0][0]))
        denorm_preds = []
        for pred in predictions:
            denorm_preds.append(self.denormalize_xy(pred[:, :2], translation, rotation))
        denorm_preds = np.array(denorm_preds)
        self.agent_gt = self.data['gt_preds'][0]
        self.all_preds = copy.deepcopy(denorm_preds)
            
    def denormalize_xy(self, xy_locations, translation=None, rotation=None):
        """Reverse the Translate and rotate operations on the input data
            Args:
                xy_locations (numpy array): XY positions for the trajectory
            Returns:
                xy_locations_normalized (numpy array): denormalized XY positions
        """
        # Apply rotation
        num = xy_locations.shape[0]
        if xy_locations.shape[0] > 1:
            trajectory = LineString(xy_locations)
        else:
            trajectory = LineString(np.concatenate(([[0.0, 0.0]], xy_locations), axis=0))

        if rotation is not None:
            trajectory = rotate(trajectory, rotation, origin=(0, 0))

        if translation is not None:
            mat = [1, 0, 0, 1, translation[0], translation[1]]
            trajectory = affine_transform(trajectory, mat)

        output = np.array(trajectory.coords, dtype=np.float32)
        if num <= 1:
            output = output[1:]

        return output

    def get_points_on_curve(self, curve_points, dists):
        """
        inputs a set of points on a curve and outputs points on that curve having distances according to dists
        :param curve_points: a list of points on a curve
        :param dists: a list of numbers indicating distances between output points
        :return: a list of points on the given curve having distances according to dists
        """
        t = np.arange(len(curve_points))
        csx = CubicSpline(t, curve_points[:, 0])
        csy = CubicSpline(t, curve_points[:, 1])
        points = [curve_points[0]]
        cur_time = 0
        for d in dists:
            dx, dy = csx(cur_time, 1), csy(cur_time, 1)
            delta_t = d / np.sqrt(dx ** 2 + dy ** 2)
            new_time = cur_time + delta_t

            left, right = cur_time, len(curve_points)
            mx_dist = np.sqrt((csx(right) - csx(cur_time)) ** 2 + (csy(right) - csy(cur_time)) ** 2)
            while mx_dist < d:
                right *= 2
                mx_dist = np.sqrt((csx(right) - csx(cur_time)) ** 2 + (csy(right) - csy(cur_time)) ** 2)

            while True:
                dist = np.sqrt((csx(new_time) - csx(cur_time)) ** 2 + (csy(new_time) - csy(cur_time)) ** 2)
                if np.abs(dist - d) <= 0.01:
                    break
                if dist > d:
                    right = new_time
                elif dist < d:
                    left = new_time
                new_time = (left + right) / 2

            cur_time = new_time
            points.append(np.array([csx(cur_time), csy(cur_time)]))
        points = np.array(points)
        return points

    def correct_gt(self, gt, scale_factor):
        """
        inputs gt points and a scale factor and scales gt points according to that scale_factor in a way that
        the output points rely on the same curve as the input points. In other words, this function only moves the
        gt points on their curve so that their distances are multiplied by the scale_factor.
        :param gt: a list of points as ground truth
        :param scale_factor: the constant we want the distance between gt points to be multiplied by
        :return:
        """
        gt_speeds = np.sqrt(((gt[1:] - gt[:-1]) ** 2).sum(1))
        return self.get_points_on_curve(gt, gt_speeds * scale_factor)

    def calc_curvature(self, x):
        """
        given any set of points x, outputs the curvature of the attack_function on those points
        """
        numerator = self.attack_function.f_zegond(x)
        denominator = (1 + self.attack_function.f_prime(x) ** 2) ** 1.5
        return numerator / denominator

    def calc_radius(self, x):
        """
        given any set of points x, outputs the radius of a circle fitting to the attack_function on those points
        """
        curv = self.calc_curvature(x)
        ret = np.zeros_like(x)
        ret[curv == 0] = 1000_000_000_000  # inf
        ret[curv != 0] = 1 / np.abs(curv[curv != 0])
        return ret

    def apply_transform_function(self, points):
        """
        Applies attack_function on the input points
        :param points: np array of points that we want to apply the transformation on
        :return: transformed points
        """
        points = points.copy()
        points[:, 1] += self.attack_function.f(points[:, 0])
        return points

    def apply_inverse_transform_function(self, points):
        """
        Applies the inverse of the transformation_function on the input points
        :param points: np array of points that we want to apply inverse transformation on
        :return: inverse transformed points
        """
        points = points.copy()
        points[:, 1] -= self.attack_function.f(points[:, 0])
        return points

    def calc_offroad(self):
        """
        Gets predictions from Uber's LaneGCN given data as input and calculates SOR and HOR.
        Also saves model predictions for ego agent.
        :return: reward of the current state as int
        """
        if self.models_name in ["LaneGCN", "MPC"]:
            # project these predictions to the space before transformation
            real_pred_points = np.matmul(self.data['rot'], (self.agent_pred - self.data['orig'].reshape(-1, 2)).T).T
            real_pred_points = self.apply_inverse_transform_function(real_pred_points)
            real_pred_points = np.matmul(self.data['rot'].T, real_pred_points.T).T + self.data['orig'].reshape(-1, 2)

            off_roads = self.am.get_raster_layer_points_boolean(real_pred_points, self.data['city'], "driveable_area")
            SOR = 1 - (off_roads.sum() / len(off_roads))
            if np.sum(off_roads) == off_roads.shape[0]:
                HOR = 0
            else:
                HOR = 1
        elif self.models_name in ["DATF", "WIMP"]:
            best_SOR, best_HOR = 0, 0
            l2_dist = 1e7
            for pred in self.all_preds:
                real_pred_points = np.matmul(self.data['rot'], (pred - self.data['orig'].reshape(-1, 2)).T).T
                real_pred_points = self.apply_inverse_transform_function(real_pred_points)
                real_pred_points = np.matmul(self.data['rot'].T, real_pred_points.T).T + self.data['orig'].reshape(-1, 2)
                off_roads = self.am.get_raster_layer_points_boolean(real_pred_points, self.data['city'], "driveable_area")
                SOR = 1 - (off_roads.sum() / len(off_roads))
                if np.sum(off_roads) == off_roads.shape[0]:
                    HOR = 0
                else:
                    HOR = 1
                dist = np.sqrt(((self.agent_gt - pred) ** 2).sum(1)).mean()
                if(dist < l2_dist):
                    best_SOR = SOR
                    best_HOR = HOR
                    l2_dist = dist
                    self.agent_pred = pred
            SOR = best_SOR
            HOR = best_HOR
        return SOR, HOR

    def attack(self, id, params, save_addr=None):
        """
        given the scenario id and transformation parameters as input, applies the corresponding transformation on the
        given scenario and calculates model's offroads on this new scenario.
        :param id: scenario id
        :param params: a dict containing transformation params
        :param save_addr: if the render flag was true when initializing the object, the address which the figure is
        going to be saved on
        :return: offroad metrics
        """
        self.attack_params = params
        self.attack_function = Combination(params)
        self.scenario_idx = id

        self.get_scenario()

        offroad = self.calc_offroad()

        if self.visualize:
            self.render(save_addr)
        return offroad

    def render(self, save_addr):
        """
        Saves the figure of the current scenario
        :return: nothing! :D
        """

        idx = self.scenario_idx

        for i, gt_pred in enumerate(self.data['gt_preds']):
            # first rotate ground truth like it is done in data for the prediction
            gt_pred[:, :] = np.matmul(self.data['rot'], (gt_pred - self.data['orig'].reshape(-1, 2)).T).T
            # skew the ground truths
            gt_pred[:, :] = self.apply_transform_function(gt_pred)
            if i == 0:  # applies gt correction given the scale_factor calculated in history correction. Note that if
                # the speed correction is off, scale_factor is equal to 1 so this won't change the gt
                gt_pred[:, :] = self.correct_gt(gt_pred, self.scale_factor)
            # undo the rotation to bring the ground truth back to the world coordinates
            gt_pred[:, :] = np.matmul(self.data['rot'].T, gt_pred.T).T + self.data['orig'].reshape(-1, 2)

        # next, make the data of scene
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = [-100.0, 100.0, -100.0, 100.0]
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(self.data['orig'][0], self.data['orig'][1], self.data['city'], radius)
        lane_ids = copy.deepcopy(lane_ids)
        lanes = dict()
        self.lane_centerlines = []
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[self.data['city']][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(self.data['rot'], (lane.centerline - self.data['orig'].reshape(-1, 2)).T).T
            centerline = self.apply_transform_function(centerline)
            self.lane_centerlines.append(centerline)

        agents_trajs_skewed = []
        for i, traj in enumerate(self.data['trajs']):
            traj = traj[:50, :]
            traj = np.matmul(self.data['rot'], (traj - self.data['orig'].reshape(-1, 2)).T).T
            traj = self.apply_transform_function(traj)
            traj_corrected = traj.copy()
            if i == 0:
                traj_corrected[:20, :] = self.correct_history(traj[:20, :])
            traj_corrected = np.matmul(self.data['rot'].T, traj_corrected.T).T + self.data['orig'].reshape(-1, 2)
            agents_trajs_skewed.append(traj_corrected)

        viz_scenario(self.lane_centerlines, agents_trajs_skewed, self.data['gt_preds'][0],
                     self.agent_pred, save_addr, self.data['rot'], self.data['orig'])
    
    def setVisualize(self, vis):
        self.visualize = vis
