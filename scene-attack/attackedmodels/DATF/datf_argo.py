
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import copy
import warnings
from shapely.geometry.polygon import Polygon
from torchvision import transforms
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from typing import Dict, List, Tuple, Optional, Union

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap


class DATF_argoverse_dataset():
    def __init__(self, maps):
        
        self.raw_dt_map_dict = {}
        self.raw_vis_map_dict = {}
        self.max_distance = 56.0
        
        for city_name in ['PIT', 'MIA']:
            self.raw_dt_map_dict[city_name] = {'map' : maps[city_name]['dt'], 'image_to_city' : maps[city_name]['image_to_city']}
        for city_name in ['PIT', 'MIA']:
                self.raw_vis_map_dict[city_name] = {'map' : maps[city_name]['mask'], 'image_to_city' : maps[city_name]['image_to_city']}
        self.context_transform = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize(-23.1, 27.3)])
        self.prior_transform = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Lambda(lambda x: torch.where(x > 0, 0.0, x)),
                                                       transforms.Lambda(lambda x: F.log_softmax(x.reshape(-1), dim=0).reshape(x.shape))])
        self.vis_map_size = (224, 224)
        self.context_map_size = (64, 64)
        self.prior_map_size = (100, 100)
                
        self.num_candidates = 12                                                    # default number of cancdidates
        
        self.decoding_steps = int(3 * 2)                                                # 3 * default sampling rate
                
        self.counter = 1
        
    def generate_trajectories(self, df, obsv_path="", pred_path=""):
          """
          Generate Trajectories w/ reference frame set to 19 (the current timestep).
          """
          
          warnings.filterwarnings("ignore") # ignore pandas copy warning.
          REFERENCE_FRAME = 19
          SEED = 88245
          
          
          # Assign Frames to Timestamps
          ts_list = df['TIMESTAMP'].unique()
          ts_list.sort()
        
          ts_mask = []
          frames = []
          for i, ts in enumerate(ts_list):
            ts_mask.append(df['TIMESTAMP'] == ts)
            frames.append(i)
        
          df.loc[:, 'FRAME'] = np.select(ts_mask, frames)
        
          # Filter TRACK_IDs that do not exist at reference_frame.
          present_agents = df[df.FRAME == REFERENCE_FRAME]
          present_mask = np.isin(df["TRACK_ID"].to_numpy(), present_agents["TRACK_ID"].to_numpy())
          df = df[present_mask]
        
          track_masks = []
          observation_timelens = []
          observation_curvelens = []
          observation_curvatures = []
          prediction_timelens = []
          prediction_curvelens = []
          prediction_curvatures = []
          for track_id in df["TRACK_ID"].unique():
            # Get trajectories corresponding to a track_id
            track_mask = (df.TRACK_ID==track_id)
            track_masks.append(track_mask)
        
            track_df = df[track_mask]
            
            track_frames = track_df["FRAME"].to_numpy() # All frame indices except those for missing frames.
            start_frame = track_frames[0]
            end_frame = track_frames[-1]
            for earliest_frame in range(REFERENCE_FRAME, start_frame-1, -1):
              """
              Find the earliest frame within the longest continuous
              observation sequence that contains reference_frame.
              """
              if earliest_frame not in track_frames:
                earliest_frame += 1
                break
              
            obsv_timelen = REFERENCE_FRAME - earliest_frame + 1
            
            obsv_track_df = track_df[(track_df.FRAME>=earliest_frame) & (track_df.FRAME<=REFERENCE_FRAME)]
            obsv_XY = obsv_track_df[['X', 'Y']].to_numpy()
            
            if len(obsv_XY.shape) == 1:
              obsv_XY = np.expand_dims(obsv_XY, axis=0)
        
            obsv_td = np.diff(obsv_XY, axis=0)
            obsv_curvlen = np.linalg.norm(obsv_td, axis=1).sum()
        
            obsv_err = obsv_XY[-1] - obsv_XY[0]
            obsv_disp = np.sqrt(obsv_err.dot(obsv_err))
        
            if obsv_disp != 0.0:
              obsv_curvature = obsv_curvlen / obsv_disp
            
            else:
              obsv_curvature = float("inf")
        
            observation_timelens.append(obsv_timelen)
            observation_curvelens.append(obsv_curvlen)
            observation_curvatures.append(obsv_curvature)
        
            pred_timelen = 0
            pred_curvlen = None
            pred_curvature = None
            if pred_path is not None:
              if REFERENCE_FRAME != end_frame:
                for latest_frame in range(REFERENCE_FRAME+1, end_frame+1):
                  """
                  Find the latest frame within the longest continuous
                  prediction sequence right next to the reference frame.
                  """
                  if latest_frame not in track_frames:
                    latest_frame -= 1
                    break
            
                pred_timelen = latest_frame - REFERENCE_FRAME
                if pred_timelen != 0:
                  pred_track_df = track_df[(track_df.FRAME>REFERENCE_FRAME) & (track_df.FRAME<=latest_frame)]
                  pred_XY = pred_track_df[['X', 'Y']].to_numpy()
                
                  if len(pred_XY.shape) == 1:
                    pred_XY = np.expand_dims(pred_XY, axis=0)
        
                  pred_td = np.diff(pred_XY, axis=0)
                  pred_curvlen = np.linalg.norm(pred_td, axis=1).sum()
        
                  pred_err = pred_XY[-1] - pred_XY[0]
                  pred_disp = np.sqrt(pred_err.dot(pred_err))
        
                  if pred_disp != 0.0:
                    pred_curvature = pred_curvlen / pred_disp
                  
                  else:
                    pred_curvature = float("inf")
        
            prediction_timelens.append(pred_timelen)
            prediction_curvelens.append(pred_curvlen)
            prediction_curvatures.append(pred_curvature)
        
          df.loc[:, 'OBSERVATION_TIMELEN'] = np.select(track_masks, observation_timelens)
          df.loc[:, 'OBSERVATION_CURVELEN'] = np.select(track_masks, observation_curvelens)
          df.loc[:, 'OBSERVATION_CURVATURE'] = np.select(track_masks, observation_curvatures)
        
          df.loc[:, 'PREDICTION_TIMELEN'] = np.select(track_masks, prediction_timelens)
          df.loc[:, 'PREDICTION_CURVELEN'] = np.select(track_masks, prediction_curvelens)
          df.loc[:, 'PREDICTION_CURVATURE'] = np.select(track_masks, prediction_curvatures)
        
          filter_condition = None
          for track_id in df["TRACK_ID"].unique():
            """
            Process observation & prediction trajectories with missing frames by...
            Filtering observation sequences with frames earlier than earliest_frame_idx.
            Filtering prediction sequences with frames later than latest_frame_idx.
            """
            track_mask = df["TRACK_ID"] == track_id
            observation_length = df[track_mask]['OBSERVATION_TIMELEN'].iloc[0]
            prediction_length = df[track_mask]['PREDICTION_TIMELEN'].iloc[0]
            
            track_condition = track_mask & (df.FRAME > REFERENCE_FRAME - observation_length)
            if pred_path is not None:
              track_condition = track_condition & (df.FRAME < REFERENCE_FRAME + prediction_length + 1)
            
            if filter_condition is not None:
              filter_condition = filter_condition | track_condition
            else:
              filter_condition = track_condition
        
          df = df[filter_condition]
        
          # Add X_CITY & Y_CITY features.
          XY_CITY = df[['X', 'Y']].to_numpy()
          df.loc[:, 'X_CITY'], df.loc[:, 'Y_CITY'] = XY_CITY[:, 0], XY_CITY[:, 1]
        
          # Center X_CIYU & Y_CITY to make X & Y.
          agent_df = df[df.OBJECT_TYPE=='AGENT'] # Agent of Interest (AoI)
          translation = agent_df[agent_df.FRAME == REFERENCE_FRAME][["X_CITY", "Y_CITY"]].to_numpy()
        
    
        
          XY_center = XY_CITY - translation
          df.loc[:, 'X'], df.loc[:, 'Y'] = XY_center[:, 0], XY_center[:, 1]
        
          # Partition the observation and prediction.
          observation = df[df["FRAME"] <= REFERENCE_FRAME]
        
          if pred_path is not None:
            prediction = df[df["FRAME"] > REFERENCE_FRAME]
            
          return observation, prediction
            
    
    
    def prepare_samples(self,
                        idx,
                        obsv_df,
                        pred_df,
                        pred_path = "",
                        min_enc_obsvlen = 6,
                        min_dec_obsvlen = 10,
                        min_dec_predlen = 30,
                        max_distance = 56,
                        sampling_interval = 5,
                        max_obsv_len = 4,
                        max_pred_len = 6,
                        multi_agent = True):

        ref_frame = 19
        
        # Select instance tokens for the encoding phase.
        encoding_df = obsv_df[obsv_df.OBSERVATION_TIMELEN >= min_dec_obsvlen]
        if not multi_agent:
            encoding_df = encoding_df[encoding_df.OBJECT_TYPE == "AGENT"]
            
        # Check agents are located within the max_distance at ref_frame.
        ref_df = encoding_df[encoding_df.FRAME == ref_frame]
        ref_location = ref_df[['X', 'Y']].to_numpy()

        mask = np.all(np.abs(ref_location) < max_distance, axis=1)
        ref_df = ref_df[mask]

        encoding_tokens = ref_df["TRACK_ID"].unique()

        decoding_agents_mask = None
        if pred_path is not None:
            # Select instance tokens for the decoding phase.
            decoding_df = pred_df[pred_df.OBSERVATION_TIMELEN >= min_dec_obsvlen]
            decoding_df = pred_df[pred_df.PREDICTION_TIMELEN >= min_dec_predlen]
        
            # Filter instance tokens that are not to be encoded.
            filter_mask = np.isin(decoding_df["TRACK_ID"], encoding_tokens)
            decoding_df = decoding_df[filter_mask]

            if not multi_agent:
                decoding_df = decoding_df[decoding_df.OBJECT_TYPE == "AGENT"]

            # Create a mask for encoding_tokens where True denotes that
            # the corresponding agent is to be decoded for prediction.
            decoding_agents_mask = np.isin(encoding_tokens, decoding_df["TRACK_ID"].unique())

            if decoding_agents_mask.sum() == 0:
                return None

        obsv_traj_list = []
        obsv_traj_len_list = []
        decode_start_pos_list = []
        decode_start_vel_list = []
        for token in encoding_tokens:
            agent_enc_df = encoding_df[encoding_df.TRACK_ID == token]

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

            obsv_traj_padded = []
            for traj, traj_len in zip(obsv_traj_list, obsv_traj_len_list):
                obsv_pad = max_obsv_len - traj_len
                traj_padded = np.pad(traj, ((0, obsv_pad), (0, 0)), mode='constant')
                obsv_traj_padded.append(traj_padded)

        if pred_path is not None:
            pred_traj_list = []
            pred_traj_len_list = []
            decoding_tokens = encoding_tokens[decoding_agents_mask]
            for token in decoding_tokens:
                agent_dec_df = decoding_df[decoding_df.TRACK_ID == token]

                pred_trajectory = agent_dec_df[['X', 'Y']].to_numpy().astype(np.float32)
                pred_trajectory = pred_trajectory[sampling_interval-1::sampling_interval] # Sample Trajectory w/ sampling interval.

                pred_len = len(pred_trajectory)
    
                pred_traj_list.append(pred_trajectory)
                pred_traj_len_list.append(pred_len)

            pred_traj_padded = []
            for traj, traj_len in zip(pred_traj_list, pred_traj_len_list):
                pred_pad = max_pred_len - traj_len
                traj_padded = np.pad(traj, ((0, pred_pad), (0, 0)), mode='constant')
                pred_traj_padded.append(traj_padded)
        
        else:
            decoding_tokens = encoding_tokens
        
        obsv_traj = np.array(obsv_traj_padded, dtype=np.float32)
        obsv_traj_len = np.array(obsv_traj_len_list, dtype=np.int64)
        decode_start_pos = np.array(decode_start_pos_list, dtype=np.float32)
        decode_start_vel = np.array(decode_start_vel_list, dtype=np.float32)

        pred_traj = pred_traj_len = None
        if pred_path is not None:
            pred_traj = np.array(pred_traj_padded, dtype=np.float32)
            pred_traj_len = np.array(pred_traj_len_list, dtype=np.int64)

        ego_df = obsv_df[obsv_df.OBJECT_TYPE == 'AGENT']
        ego_ref_df = ego_df[ego_df.FRAME == ref_frame]

        city_name = ego_ref_df['CITY_NAME'].values.item()
        translation = ego_ref_df[['X_CITY', 'Y_CITY']].to_numpy().squeeze()
        # yaw = ego_ref_df['ANGLE'].values.item()
        # yaw_angle = Quaternion.to_degrees(yaw)

        metadata = {'scene': str(idx),
                    'city_name': city_name,
                    'ref_translation': translation,
                    'encoding_tokens': encoding_tokens,
                    'decoding_tokens': decoding_tokens}

        return [obsv_traj], obsv_traj_len, [pred_traj], [pred_traj_len], decoding_agents_mask, [decode_start_pos], [decode_start_vel], metadata


    def getmaps(self, metadata, maps):
        
        for city_name in ['PIT', 'MIA']:
            self.raw_dt_map_dict[city_name] = {'map' : maps[city_name]['dt'], 'image_to_city' : maps[city_name]['image_to_city']}
        for city_name in ['PIT', 'MIA']:
                self.raw_vis_map_dict[city_name] = {'map' : maps[city_name]['mask'], 'image_to_city' : maps[city_name]['image_to_city']}
        self.context_transform = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize(-23.1, 27.3)])
        self.prior_transform = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Lambda(lambda x: torch.where(x > 0, 0.0, x)),
                                                       transforms.Lambda(lambda x: F.log_softmax(x.reshape(-1), dim=0).reshape(x.shape))])
        
        

        context_map = None
        prior_map = None
        vis_map = None
        if (self.vis_map_size is not None) or (self.context_map_size is not None) or (self.prior_map_size is not None):
            city_name = metadata['city_name']
            X, Y = metadata['ref_translation']

            if (self.context_map_size is not None) or (self.prior_map_size is not None):
                raw_dt_map = self.raw_dt_map_dict[city_name]['map']
                image_to_city = self.raw_dt_map_dict[city_name]['image_to_city']
                
                scale_dt_h = image_to_city[1, 1]
                translate_dt_h = image_to_city[1, 2]
                scale_dt_w = image_to_city[0, 0]
                translate_dt_w = image_to_city[0, 2]

                pixel_dims_h = scale_dt_h * self.max_distance * 2
                pixel_dims_w = scale_dt_w * self.max_distance * 2

                crop_dims_h = np.ceil(np.sqrt(2 * pixel_dims_h**2) / 10) * 10
                crop_dims_w = np.ceil(np.sqrt(2 * pixel_dims_w**2) / 10) * 10
                
                # Corners of the crop in the raw image's coordinate system.
                crop_box = (X+translate_dt_w,
                            Y+translate_dt_h,
                            crop_dims_h,
                            crop_dims_w)
                crop_patch = self.get_patch(crop_box, patch_angle=0.0)
                
                # Do Crop
                dt_crop, crop_boundary = self.crop_image(raw_dt_map,
                                                        crop_patch)
                
                
                # Corners of the final image in the crop image's coordinate system.
                final_box = (scale_dt_w*X - crop_boundary['left'] + translate_dt_w,
                            scale_dt_h*Y - crop_boundary['up'] + translate_dt_h,
                            pixel_dims_h,
                            pixel_dims_w)

                final_patch_angle = 0.0
                final_patch = self.get_patch(final_box, patch_angle=final_patch_angle)
                final_coords_in_crop = np.array(final_patch.exterior.coords)
                dt_corner_points = final_coords_in_crop[:4]
                
                
            if self.vis_map_size is not None:
                raw_vis_map = self.raw_vis_map_dict[city_name]['map']
                image_to_city = self.raw_vis_map_dict[city_name]['image_to_city']

                scale_vis_h = image_to_city[1, 1]
                translate_vis_h = image_to_city[1, 2]
                scale_vis_w = image_to_city[0, 0]
                translate_vis_w = image_to_city[0, 2]

                pixel_dims_h = scale_vis_h * self.max_distance * 2
                pixel_dims_w = scale_vis_w * self.max_distance * 2

                crop_dims_h = np.ceil(np.sqrt(2 * pixel_dims_h**2) / 10) * 10
                crop_dims_w = np.ceil(np.sqrt(2 * pixel_dims_w**2) / 10) * 10
                
                # Corners of the crop in the raw image's coordinate system.
                crop_box = (X+translate_vis_w,
                            Y+translate_vis_h,
                            crop_dims_h,
                            crop_dims_w)
                crop_patch = self.get_patch(crop_box, patch_angle=0.0)

                # Do Crop
                vis_crop, crop_boundary = self.crop_image(raw_vis_map,
                                                          crop_patch)
                
                # Corners of the final image in the crop image's coordinate system.
                final_box = (scale_vis_w*X - crop_boundary['left'] + translate_vis_w,
                            scale_vis_h*Y - crop_boundary['up'] + translate_vis_h,
                            pixel_dims_h,
                            pixel_dims_w)
                
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

        return [context_map], [prior_map], vis_map
    
    def collate(self, input_params):
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
        # Typically, Num obv agents in batch_i < Num pred agents in batch_i 
            
        obsv_traj, obsv_traj_len, pred_traj, pred_traj_len, decoding_agents_mask, decode_start_pos, decode_start_vel, context_map, prior_map, vis_map, metadata = input_params
    
    
    
        # Observation trajectories
        num_obsv_agents = np.array([len(x) for x in obsv_traj])
        obsv_traj = np.concatenate(obsv_traj, axis=0)
    
        # Convert to Tensor
        num_obsv_agents = torch.LongTensor(num_obsv_agents)
        obsv_traj = torch.FloatTensor(obsv_traj)
        obsv_traj_len = torch.LongTensor(obsv_traj_len)
    
        # Prediction trajectories
        condition = [isinstance(element, np.ndarray) for element in pred_traj]
        if False not in condition:
            pred_traj = np.concatenate(pred_traj, axis=0)
            pred_traj = torch.FloatTensor(pred_traj)
    
        condition = [isinstance(element, np.ndarray) for element in pred_traj_len]
        if False not in condition:
            num_pred_agents = np.array([len(pred_traj)])
            pred_traj_len = np.concatenate(pred_traj_len, axis=0)
            num_pred_agents = torch.LongTensor(num_pred_agents)
            pred_traj_len = torch.LongTensor(pred_traj_len)
        else:
            num_pred_agents = tuple(None for _ in range(len(pred_traj_len)))
    
        # Decoding agent mask
        condition = [isinstance(element, np.ndarray) for element in decoding_agents_mask]
        if False not in condition:
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
    
    
    def transform_image(self, image,
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
    
    
    def crop_image(self, image,
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
    
    def get_patch(self, patch_box: Tuple[float, float, float, float],
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
    
    def write(self, obj, name):
        cv2.imwrite(name + '.jpg', obj)
    
    
    def get_model(self):
        
        scene_distance=56.0
        velocity_const=0.5
        motion_features=128
        rnn_layers=1
        rnn_dropout=0
        detach_output = 0
    
        from attackedmodels.DATF.Proposed.models import AttGlobal_Scene_CAM_NFDecoder
        from attackedmodels.DATF.Proposed.utils import ModelTrainer

        model = AttGlobal_Scene_CAM_NFDecoder(scene_distance=scene_distance,
                                              velocity_const=velocity_const,
                                              motion_features=motion_features,
                                              rnn_layers=rnn_layers,
                                              rnn_dropout=rnn_dropout,
                                              detach_output=bool(detach_output))
        self.context_map_size = (64, 64)

        return model
        
            
    def run_model(self, model, model_input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model.eval()

        
        with torch.no_grad():
                H = W = 64
                coordinate_2d = np.indices((H, W))
                coordinate = np.ravel_multi_index(coordinate_2d, dims=(H, W))
                coordinate = torch.FloatTensor(coordinate)
                coordinate = coordinate.reshape((1, 1, H, W))
    
                coordinate_std, coordinate_mean = torch.std_mean(coordinate)
                coordinate = (coordinate - coordinate_mean) / coordinate_std
    
                distance_2d = coordinate_2d - np.array([(H-1)/2, (H-1)/2]).reshape((2, 1, 1))
                distance = np.sqrt((distance_2d ** 2).sum(axis=0))
                distance = torch.FloatTensor(distance)
                distance = distance.reshape((1, 1, H, W))
    
                distance_std, distance_mean = torch.std_mean(distance)
                distance = (distance - distance_mean) / distance_std
            
                coordinate = coordinate.to(device)
                distance = distance.to(device)
                
                
                obsv_traj, obsv_traj_len, obsv_num_agents, \
                    pred_traj, pred_traj_len, pred_num_agents, \
                    obsv_to_pred_mask, init_pos, init_vel, \
                    context_map, _, vis_map, metadata = model_input
                    
                
                # Detect dynamic batch size
                batch_size = obsv_num_agents.size(0)
                obsv_total_agents = obsv_traj.size(0)
                pred_total_agents = pred_traj.size(0)
    
                obsv_traj = obsv_traj.to(device)
                # obsv_traj_len = obsv_traj_len.to(self.device)
                # obsv_num_agents = obsv_num_agents.to(self.device)
    
                pred_traj = pred_traj.to(device)
                # pred_traj_len = pred_traj_len.to(self.device)
                # pred_num_agents = pred_num_agents.to(self.device)
    
                # obsv_to_pred_mask = obsv_to_pred_mask.to(device)
    
                init_pos = init_pos.to(device)
                init_vel = init_vel.to(device)
    
                # Cat coordinates and distances info to the context map.
                coordinate_batch = coordinate.repeat(batch_size, 1, 1, 1)
                distance_batch = distance.repeat(batch_size, 1, 1, 1)
                context_map = torch.cat((context_map.to(device), coordinate_batch, distance_batch), dim=1)
                   
                pred_traj_ = pred_traj.unsqueeze(1)
                scene_idx = torch.arange(batch_size).repeat_interleave(pred_num_agents)
                gen_traj, _, _, _, _, _ = self.model(obsv_traj,
                                                     obsv_traj_len,
                                                     obsv_num_agents,
                                                     init_pos,
                                                     init_vel,
                                                     self.decoding_steps,                                
                                                     self.num_candidates,                               
                                                     context_map,
                                                     scene_idx,
                                                     obsv_to_pred_mask)
    
        
        return gen_traj
    
    





            
            