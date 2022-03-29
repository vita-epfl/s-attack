import pathlib
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Union

from compress_pickle import dump, load
from pyquaternion import Quaternion
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box

class ParallelSim(object):
    def __init__(self, processes):
        self.pool = mp.Pool(processes=processes)
        self.total_processes = 0
        self.completed_processes = 0
        self.results = []

    def add(self, func, args):
        self.pool.apply_async(func=func, args=args, callback=self.complete)
        self.total_processes += 1

    def complete(self, result):
        if result is not None:
            self.results.append(result)
            self.completed_processes += 1

            if self.completed_processes == self.total_processes:
                print('-- loaded {:d}/{:d}, complete.'.format(self.completed_processes,
                                                              self.total_processes))
            else:
                print('-- loaded {:d}/{:d}'.format(self.completed_processes,
                                                    self.total_processes), end='\r')

    def run(self):
        self.pool.close()
        self.pool.join()

    def get_results(self):
        return self.results

def nuscenes_collate(batch):
    # batch_i:
    # 1. past_agents_traj : (Num obv agents in batch_i X 20 X 2)
    # 2. past_agents_traj_len : (Num obv agents in batch_i, )
    # 3. future_agents_traj : (Num pred agents in batch_i X 20 X 2)
    # 4. future_agents_traj_len : (Num pred agents in batch_i, )
    # 5. future_agent_masks : (Num obv agents in batch_i)
    # 6. decode_rel_pos: (Num pred agents in batch_i X 2)
    # 7. decode_start_pos: (Num pred agents in batch_i X 2)
    # 8. map_image : (3 X 224 X 224)
    # 9. scene ID: (string)
    # Typically, Num obv agents in batch_i < Num pred agents in batch_i ##

    batch_size = len(batch)
    obsv_traj, obsv_traj_len, pred_traj, pred_traj_len, decoding_agents_mask, decode_start_pos, decode_start_vel, context_map, prior_map, vis_map, metadata = list(zip(*batch))

    # Observation trajectories
    num_obsv_agents = np.array([len(x) for x in obsv_traj_len])
    obsv_traj = np.concatenate(obsv_traj, axis=0)
    obsv_traj_len = np.concatenate(obsv_traj_len, axis=0)

    # Convert to Tensor
    num_obsv_agents = torch.LongTensor(num_obsv_agents)
    obsv_traj = torch.FloatTensor(obsv_traj)
    obsv_traj_len = torch.LongTensor(obsv_traj_len)

    # Prediction trajectories
    num_pred_agents = np.array([len(x) for x in pred_traj_len])
    pred_traj = np.concatenate(pred_traj, axis=0)
    pred_traj_len = np.concatenate(pred_traj_len, axis=0)

    # Convert to Tensor
    num_pred_agents = torch.LongTensor(num_pred_agents)
    pred_traj = torch.FloatTensor(pred_traj)
    pred_traj_len = torch.LongTensor(pred_traj_len)

    # Decoding agent mask
    decoding_agents_mask = np.concatenate(decoding_agents_mask, axis=0)
    decoding_agents_mask = torch.BoolTensor(decoding_agents_mask)

    # Decode start vel & pos
    decode_start_vel = np.concatenate(decode_start_vel, axis=0)
    decode_start_pos = np.concatenate(decode_start_pos, axis=0)
    decode_start_vel = torch.FloatTensor(decode_start_vel)
    decode_start_pos = torch.FloatTensor(decode_start_pos)

    condition = [isinstance(element, torch.Tensor) for element in context_map]
    if False not in condition:
        context_map = torch.stack(context_map, dim=0)

    condition = [isinstance(element, torch.Tensor) for element in prior_map]
    if False not in condition:
        prior_map = torch.stack(prior_map, dim=0)
    
    data = (
        obsv_traj, obsv_traj_len, num_obsv_agents,
        pred_traj, pred_traj_len, num_pred_agents, 
        decoding_agents_mask, decode_start_pos, decode_start_vel, 
        context_map, prior_map, vis_map, metadata
    )

    return data

class NuscenesDataset(Dataset):	
    def __init__(self, data_dir, data_partition, logger, sampling_rate, intrinsic_rate=2,
                 sample_stride=1, max_distance=56.0, num_workers=None, cache_file=None,
                 context_map_size=None, prior_map_size=None, vis_map_size=None, multi_agent=True):
        """
        data_dir: Dataset root directory
        data_parititon: Dataset Parition (train | val | test_obs)
        sampling_rate: Physical sampling rate of processed trajectory (Hz)
        intrinsic_rate: Physical sampling rate of raw trajectory (Hz, eg., Argo:10, nuScenes:2)
        sample_stride: The interval between the reference frames in a single episode
        min_past_obv_len: Minimum length of the agent's past trajectory to encode
        min_future_obv_len: Minimum length of the agent's past trajectory to decode
        min_future_pred_len: Minimum length of the agent's future trajectory to decode
        max_distance: Maximum physical distance from the ROI center to an agent's current position
        multi_agent: Multi-agent experiment.
        """
        super(NuscenesDataset, self).__init__()
        self.logger = logger

        self.data_dir = data_dir
        self.data_partition = data_partition

        if num_workers:
            self.num_workers = num_workers
        else:
            self.num_workers = mp.cpu_count()

        # Sampling Interval = "intrinsic sampling rate" / sampling rate
        self.intrinsic_rate = intrinsic_rate
        if intrinsic_rate % sampling_rate:
            raise ValueError("Intrinsic sampling rate must be evenly divisble by sampling rate.\n Intrinsic SR: {:d}, Given SR: {:d}".format(10, sampling_rate))
        self.sampling_interval = int(self.intrinsic_rate // sampling_rate)

        self.max_obsv_len = int(self.intrinsic_rate * 2 // self.sampling_interval)
        self.max_pred_len = int(self.intrinsic_rate * 3 // self.sampling_interval)

        self.sample_stride = sample_stride
        self.min_enc_obsvlen = self.sampling_interval + 1
        self.min_dec_obsvlen = int(1 * self.intrinsic_rate)	
        self.min_dec_predlen = int(3 * self.intrinsic_rate)
        self.max_distance = max_distance

        self.multi_agent = multi_agent

        self.vis_map_size = vis_map_size
        self.context_map_size = context_map_size
        self.prior_map_size = prior_map_size

        if (self.context_map_size is not None) or (self.prior_map_size is not None):
            self.raw_dt_map_dict = {}
            for city_name in ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']:
                raw_dt_map_path = pathlib.Path(self.data_dir).joinpath('raw_map', '{:s}_dt.pkl'.format(city_name))
                self.raw_dt_map_dict[city_name] = load(raw_dt_map_path)

        if self.vis_map_size is not None:
            self.raw_vis_map_dict = {}
            for city_name in ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']:
                raw_vis_map_path = pathlib.Path(self.data_dir).joinpath('raw_map', '{:s}_mask.pkl'.format(city_name))
                self.raw_vis_map_dict[city_name] = load(raw_vis_map_path)


        if self.context_map_size is not None:
            self.context_transform = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize(-23.3, 25.3)])

        if self.prior_map_size is not None:
            self.prior_transform = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Lambda(lambda x: torch.where(x > 0, 0.0, x)),
                                                       transforms.Lambda(lambda x: F.log_softmax(x.reshape(-1), dim=0).reshape(x.shape))])

        if cache_file is None:
            self.load_data()
        else:
            cache_path = pathlib.Path(cache_file)

            if cache_path.is_file():
                self.load_cache(cache_path)
            else:
                self.load_data(cache_path=cache_path)
        
        self.logger.info('Data Loading Complete!')

    def __getitem__(self, idx):
        if type(idx) is str:
            idx = self.get_data_idx(idx)

        obsv_traj = self.obsv_traj[idx]
        obsv_traj_len = self.obsv_traj_len[idx]
        pred_traj = self.pred_traj[idx]
        pred_traj_len = self.pred_traj_len[idx]
        decoding_agents_mask = self.decoding_agents_mask[idx]
        decode_start_pos = self.decode_start_pos[idx]
        decode_start_vel = self.decode_start_vel[idx]
        metadata = self.metadata[idx]

        context_map = None
        prior_map = None
        vis_map = None
        if (self.vis_map_size is not None) or (self.context_map_size is not None) or (self.prior_map_size is not None):
            city_name = metadata['city_name']
            X, Y = metadata['ref_translation']
            ref_angle = metadata['ref_angle']

            if (self.context_map_size is not None) or (self.prior_map_size is not None):
                raw_dt_map = self.raw_dt_map_dict[city_name]['map']
                scale_dt_h = self.raw_dt_map_dict[city_name]['scale_h']
                scale_dt_w = self.raw_dt_map_dict[city_name]['scale_w']
        
                pixel_dims_h = scale_dt_h * self.max_distance * 2
                pixel_dims_w = scale_dt_w * self.max_distance * 2

                crop_dims_h = np.ceil(np.sqrt(2 * pixel_dims_h**2) / 10) * 10
                crop_dims_w = np.ceil(np.sqrt(2 * pixel_dims_w**2) / 10) * 10
                
                # Corners of the crop in the raw image's coordinate system.
                crop_box = (scale_dt_w*X,
                            scale_dt_h*Y,
                            crop_dims_h,
                            crop_dims_w)
                crop_patch = self.get_patch(crop_box, patch_angle=0.0)

                # Do Crop
                dt_crop, crop_boundary = self.crop_image(raw_dt_map,
                                                        crop_patch)
                
                # Corners of the final image in the crop image's coordinate system.
                final_box = (scale_dt_w*X - crop_boundary['left'],
                            scale_dt_h*Y - crop_boundary['up'],
                            pixel_dims_h,
                            pixel_dims_w)
                
                # final_patch_angle = ref_angle
                final_patch_angle = 0.0
                final_patch = self.get_patch(final_box, patch_angle=final_patch_angle)
                final_coords_in_crop = np.array(final_patch.exterior.coords)
                dt_corner_points = final_coords_in_crop[:4]

            if self.vis_map_size is not None:
                raw_vis_map = self.raw_vis_map_dict[city_name]['map']
                scale_vis_h = self.raw_vis_map_dict[city_name]['scale_h']
                scale_vis_w = self.raw_vis_map_dict[city_name]['scale_w']

                pixel_dims_h = scale_vis_h * self.max_distance * 2
                pixel_dims_w = scale_vis_w * self.max_distance * 2

                crop_dims_h = np.ceil(np.sqrt(2 * pixel_dims_h**2) / 10) * 10
                crop_dims_w = np.ceil(np.sqrt(2 * pixel_dims_w**2) / 10) * 10
                
                # Corners of the crop in the raw image's coordinate system.
                crop_box = (scale_vis_w*X,
                            scale_vis_h*Y,
                            crop_dims_h,
                            crop_dims_w)
                crop_patch = self.get_patch(crop_box, patch_angle=0.0)

                # Do Crop
                vis_crop, crop_boundary = self.crop_image(raw_vis_map,
                                                         crop_patch)
                
                # Corners of the final image in the crop image's coordinate system.
                final_box = (scale_vis_w*X - crop_boundary['left'],
                             scale_vis_h*Y - crop_boundary['up'],
                             pixel_dims_h,
                             pixel_dims_w)
                
                # final_patch_angle = ref_angle
                final_patch_angle = 0.0
                final_patch = self.get_patch(final_box, patch_angle=final_patch_angle)
                final_coords_in_crop = np.array(final_patch.exterior.coords)
                vis_corner_points = final_coords_in_crop[:4]

        if self.vis_map_size is not None:
            vis_map = self.transform_image(vis_crop.copy(),
                                           vis_corner_points,
                                           self.vis_map_size)

        if self.context_map_size is not None:
            context_map = self.transform_image(dt_crop.copy(),
                                               dt_corner_points,
                                               self.context_map_size,
                                               self.context_transform)
            context_map = context_map.float()

        if self.prior_map_size is not None:
            prior_map = self.transform_image(dt_crop.copy(),
                                             dt_corner_points,
                                             self.prior_map_size,
                                             self.prior_transform)
            prior_map = prior_map.float()

        episode = (obsv_traj, obsv_traj_len, pred_traj, pred_traj_len, decoding_agents_mask, decode_start_pos, decode_start_vel, context_map, prior_map, vis_map, metadata)
        return episode

    def __len__(self):
        return len(self.scene)

    @staticmethod
    def get_patch(patch_box: Tuple[float, float, float, float],
                  patch_angle: float = 0.0) -> Polygon:
        """
        Convert patch_box to shapely Polygon coordinates.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :return: Box Polygon for patch_box.
        """
        patch_x, patch_y, patch_h, patch_w = patch_box

        x_min = patch_x - patch_w / 2.0
        y_min = patch_y - patch_h / 2.0
        x_max = patch_x + patch_w / 2.0
        y_max = patch_y + patch_h / 2.0

        patch = box(x_min, y_min, x_max, y_max)
        patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

        return patch

    @staticmethod
    def crop_image(image,
                   crop_patch):

        image_h, image_w = image.shape[:2]

        # Corners of the crop in the raw image's coordinate system.
        crop_coords_in_raw = np.array(crop_patch.exterior.coords)

        _, lower_right_corner, _, upper_left_corner = crop_coords_in_raw[:4]
        crop_left, crop_up = upper_left_corner
        crop_right, crop_down = lower_right_corner

        if crop_left < 0:
            pad_left = int(-crop_left) + 1
            crop_left = 0
        
        else:
            pad_left = 0
            crop_left = int(crop_left)
        
        if crop_right >= image_w + 1:
            pad_right = int(crop_right - image_w)
            crop_right = int(image_w)
        else:
            pad_right = 0
            crop_right = int(crop_right)
        
        if crop_up < 0:
            pad_up = int(-crop_up) + 1
            crop_up = 0
        else:
            pad_up = 0
            crop_up = int(crop_up)

        if crop_down >= image_h + 1:
            pad_down = int(crop_down - image_h)
            crop_down = int(image_h)
        else:
            pad_down = 0
            crop_down = int(crop_down)

        image_crop = image[crop_up:crop_down,
                           crop_left:crop_right].copy()

        if pad_left or pad_right or pad_up or pad_down:
            image_crop = np.pad(image_crop, ((pad_up, pad_down),(pad_left, pad_right)), mode='constant')
            
        crop_boundary = {'up': crop_up-pad_up,
                         'down': crop_down+pad_down,
                         'left': crop_left-pad_left,
                         'right': crop_right+pad_right}

        return image_crop, crop_boundary

    @staticmethod
    def transform_image(image,
                        from_corner_points,
                        to_image_size,
                        torch_transform=None):
        to_corner_points = np.array([[to_image_size[1], 0],
                                     [to_image_size[1], to_image_size[0]],
                                     [0, to_image_size[0]],
                                     [0, 0]])
        
        transform_matrix = cv2.getPerspectiveTransform(from_corner_points.astype(np.float32),
                                                       to_corner_points.astype(np.float32))
        
        transformed = cv2.warpPerspective(image,
                                          transform_matrix,
                                          (to_image_size[1], to_image_size[0]))
        
        if torch_transform is not None:
            transformed = torch_transform(transformed)
        
        return transformed

    def get_scene_name(self, idx):
        if type(idx) is str:
            idx = self.get_data_idx(idx)

        return self.scene[idx]
    
    def get_data_idx(self, scene):
        return self.scene_to_idx_dict[scene]

    def load_cache(self, cache_path):
        results = load(cache_path)
        self.logger.info('Found {:s} set cache {:s} with {:d} samples.'.format(self.data_partition, str(cache_path), len(results[0])))

        self.obsv_traj = results[0]
        self.obsv_traj_len = results[1]
        self.pred_traj = results[2]
        self.pred_traj_len = results[3]
        self.decoding_agents_mask = results[4]
        self.decode_start_pos = results[5]
        self.decode_start_vel = results[6]
        self.metadata = results[7]

        self.scene = [metadata['scene'] for metadata in results[7]]
        self.scene_to_idx_dict = {key: val for val, key in enumerate(self.scene)}

    def load_data(self, cache_path=None):
        data_path = pathlib.Path(self.data_dir)

        raw_map_path = data_path.joinpath('raw_map')
        
        partition_path = data_path.joinpath(self.data_partition)
        obsv_path = partition_path.joinpath('observation')
        pred_path= partition_path.joinpath('prediction')

        obsv_sample_paths = [file_path for file_path in obsv_path.glob("*.pkl")]
        obsv_sample_paths.sort()
        pred_sample_paths = [file_path for file_path in pred_path.glob("*.pkl")]
        pred_sample_paths.sort()

        if len(obsv_sample_paths) != len(pred_sample_paths):
            msg = "# of files for observation and prediction are different.\n"
            msg += "observation dir: {:s}, {:d} files.\n".format(str(obsv_path), len(obsv_sample_paths))
            msg += "prediction dir: {:s}, {:d} files.".format(str(pred_path), len(pred_sample_paths))
            raise ValueError(msg)

        self.logger.info('Found {:d} {:s} set samples.'.format(len(obsv_sample_paths), self.data_partition))
        runner = ParallelSim(processes=self.num_workers)
        fixed_args = (self.min_enc_obsvlen, self.min_dec_obsvlen, self.min_dec_predlen,
                      self.max_distance, self.sampling_interval, self.max_obsv_len,
                      self.max_pred_len, self.multi_agent)
        
        for obsv_path, pred_path in zip(obsv_sample_paths, pred_sample_paths):
            args = (obsv_path, pred_path) + fixed_args
            runner.add(self.prepare_samples, args)
        
        runner.run()
        results = runner.get_results()
        results.sort(key=lambda sample: sample[-1]['scene']) # sort by scene_name.
        results = list(zip(*results)) # transpose data structure.

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dump(results, cache_path)

        self.obsv_traj = results[0]
        self.obsv_traj_len = results[1]
        self.pred_traj = results[2]
        self.pred_traj_len = results[3]
        self.decoding_agents_mask = results[4]
        self.decode_start_pos = results[5]
        self.decode_start_vel = results[6]
        self.metadata = results[7]
        
        self.scene = [metadata['scene'] for metadata in results[7]]
        self.scene_to_idx_dict = {key: val for val, key in enumerate(self.scene)}

    @staticmethod
    def prepare_samples(obsv_path: pathlib.Path,
                        pred_path: pathlib.Path,
                        min_enc_obsvlen: int,
                        min_dec_obsvlen: int,
                        min_dec_predlen: int,
                        max_distance: float,
                        sampling_interval: int,
                        max_obsv_len: int,
                        max_pred_len: int,
                        multi_agent: bool = True):

        obsv_scene_name = obsv_path.stem
        pred_scene_name = pred_path.stem

        if (obsv_scene_name != pred_scene_name):
            msg = "Observation and Prediction samples do not match.\n"
            msg += "observation file: {:s}\n".format(str(obsv_path))
            msg += "prediction file: {:s}".format(str(pred_path))
            raise ValueError(msg)

        _, ref_frame = [int(string) for string in obsv_scene_name.split('-')]

        obsv_df = load(obsv_path)
        pred_df = load(pred_path)

        # Select instance tokens for the encoding phase.
        encoding_df = obsv_df[obsv_df.OBSERVATION_TIMELEN >= min_dec_obsvlen]
        if not multi_agent:
            encoding_df = encoding_df[encoding_df.OBJECT_ATTRIBUTE == "ego"]
            
        # Check agents are located within the max_distance at ref_frame.
        ref_df = encoding_df[encoding_df.FRAME == ref_frame]
        ref_location = ref_df[['X', 'Y']].to_numpy()

        mask = np.all(np.abs(ref_location) < max_distance, axis=1)
        ref_df = ref_df[mask]

        encoding_tokens = ref_df["INSTANCE_TOKEN"].unique()

        # Select instance tokens for the decoding phase.
        decoding_df = pred_df[~pred_df.OBJECT_CATEGORY.str.contains("human|bicycle")]
        decoding_df = decoding_df[~decoding_df.OBJECT_ATTRIBUTE.str.contains("parked")]
        decoding_df = decoding_df[decoding_df.OBSERVATION_TIMELEN >= min_dec_obsvlen]
        decoding_df = decoding_df[decoding_df.PREDICTION_TIMELEN >= min_dec_predlen]
        
        # Filter instance tokens that are not to be encoded.
        filter_mask = np.isin(decoding_df["INSTANCE_TOKEN"], encoding_tokens)
        decoding_df = decoding_df[filter_mask]

        if not multi_agent:
            decoding_df = decoding_df[decoding_df.OBJECT_ATTRIBUTE == "ego"]

        # Create a mask for encoding_tokens where True denotes that
        # the corresponding agent is to be decoded for prediction.
        decoding_agents_mask = np.isin(encoding_tokens, decoding_df["INSTANCE_TOKEN"].unique())

        if decoding_agents_mask.sum() == 0:
            return None

        obsv_traj_list = []
        obsv_traj_len_list = []
        decode_start_pos_list = []
        decode_start_vel_list = []
        for token in encoding_tokens:
            agent_enc_df = encoding_df[encoding_df.INSTANCE_TOKEN == token]

            obsv_trajectory = agent_enc_df[['X', 'Y']].to_numpy().astype(np.float32)
            obsv_trajectory = obsv_trajectory[-1::-sampling_interval, :][::-1, :] # Sample Trajectory w/ sampling interval.

            obsv_len = len(obsv_trajectory)
            
            decode_start_pos = obsv_trajectory[-1]
            decode_start_vel = 0.0
            if obsv_len > 1:
                decode_start_vel = (obsv_trajectory[-1] - obsv_trajectory[-2]) # decode velocity

            obsv_traj_list.append(obsv_trajectory)
            obsv_traj_len_list.append(obsv_len)
            decode_start_pos_list.append(decode_start_pos)
            decode_start_vel_list.append(decode_start_vel)

        pred_traj_list = []
        pred_traj_len_list = []
        decoding_tokens = encoding_tokens[decoding_agents_mask]
        for token in decoding_tokens:
            agent_dec_df = decoding_df[decoding_df.INSTANCE_TOKEN == token]

            pred_trajectory = agent_dec_df[['X', 'Y']].to_numpy().astype(np.float32)
            pred_trajectory = pred_trajectory[sampling_interval-1::sampling_interval] # Sample Trajectory w/ sampling interval.

            pred_len = len(pred_trajectory)
 
            pred_traj_list.append(pred_trajectory)
            pred_traj_len_list.append(pred_len)

        obsv_traj_padded = []
        for traj, traj_len in zip(obsv_traj_list, obsv_traj_len_list):
           obsv_pad = max_obsv_len - traj_len
           traj_padded = np.pad(traj, ((0, obsv_pad), (0, 0)), mode='constant')
           obsv_traj_padded.append(traj_padded)

        pred_traj_padded = []
        for traj, traj_len in zip(pred_traj_list, pred_traj_len_list):
           pred_pad = max_pred_len - traj_len
           traj_padded = np.pad(traj, ((0, pred_pad), (0, 0)), mode='constant')
           pred_traj_padded.append(traj_padded)
        
        obsv_traj = np.array(obsv_traj_padded, dtype=np.float32)
        obsv_traj_len = np.array(obsv_traj_len_list, dtype=np.int64)
        decode_start_pos = np.array(decode_start_pos_list, dtype=np.float32)
        decode_start_vel = np.array(decode_start_vel_list, dtype=np.float32)

        pred_traj = np.array(pred_traj_padded, dtype=np.float32)
        pred_traj_len = np.array(pred_traj_len_list, dtype=np.int64)

        ego_df = obsv_df[obsv_df.OBJECT_ATTRIBUTE == 'ego']
        ego_ref_df = ego_df[ego_df.FRAME == ref_frame]

        city_name = ego_ref_df['CITY_NAME'].values.item()
        translation = ego_ref_df[['X_CITY', 'Y_CITY']].to_numpy().squeeze()
        rotation = ego_ref_df[['QW','QX','QY','QZ']].to_numpy().squeeze()
        yaw, _, _ = Quaternion(rotation).yaw_pitch_roll
        yaw_angle = Quaternion.to_degrees(yaw)

        metadata = {'scene': obsv_scene_name,
                    'city_name': city_name,
                    'ref_translation': translation,
                    'ref_angle': yaw_angle,
                    'encoding_tokens': encoding_tokens,
                    'decoding_tokens': decoding_tokens}

        return obsv_traj, obsv_traj_len, pred_traj, pred_traj_len, decoding_agents_mask, decode_start_pos, decode_start_vel, metadata


class dummy_logger(object):
    def info(self, x):
        print(x)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', default=None, help="")
    args = parser.parse_args()

    logger = dummy_logger()
    dataset = NuscenesDataset('./data/Preprocessed/nuScenes', 'val', sampling_rate=2, logger=logger,
                              sample_stride=1, context_map_size=(64, 64), prior_map_size=(100, 100), vis_map_size=(224, 224),
                              max_distance=56, cache_file=args.cache, multi_agent=True)
    
    loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True,
                        collate_fn=nuscenes_collate,
                        num_workers=0)
    
    for batch in loader:
        obsv_traj, obsv_traj_len, obsv_num_agents, \
        pred_traj, pred_traj_len, pred_num_agents, \
        obsv_to_pred_mask, init_pos, init_vel, \
        context_map, prior_map, vis_map, sample_code = batch
        
        import pdb; pdb.set_trace()
        pass