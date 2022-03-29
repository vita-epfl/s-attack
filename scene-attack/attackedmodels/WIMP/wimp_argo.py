import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import re
from collections import OrderedDict
from scipy.spatial.distance import cdist
from shapely.affinity import affine_transform, rotate
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union # NOQA
from collections import defaultdict
import collections.abc as container_abcs
from torch._six import string_classes
from itertools import zip_longest


from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import (
    filter_candidate_centerlines,
    get_centerlines_most_aligned_with_trajectory,
    remove_overlapping_lane_seq,
)

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import cascaded_union

from argoverse.utils.centerline_utils import (
    get_nt_distance,
)
from argoverse.utils.mpl_plotting_utils import visualize_centerline

import sys
from attack_functions import Combination

class WIMP_argoverse_dataset():
    def __init__(self, argo_path, mode = "val"):
        self.argo_path = argo_path
        self.avm = ArgoverseMap()
        self.mfu = MapFeaturesUtils()
        self.is_oracle = False
        self.heuristic = True
        self.heuristic_str = "HEURISTIC_" if self.heuristic else ""
        self.mode = mode
        self.outsteps = 30
        self.delta = False
        self.delta_str = "_delta" if self.delta else ""
        self.max_social_agents = 30
        self.timesteps = 20
        self.str = "_PARTIAL"
        self.map_features_flag = False
        self.social_features_flag = True
        self.show_speed_correction = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        
    def preprocess(self, idx, orig, rot, attack_params, file_name):
        self.orig = orig
        self.rot = rot
        self.attack_params = attack_params
        self.attack_function = Combination(self.attack_params)
        input_param = {'path': file_name, 'xy_features_flag': True,
                       'xy_features_normalize_flag': True,
                       'map_features_flag': True,
                       'social_features_flag': True, 'avm': self.avm, 'mfu': self.mfu,
                       'timesteps': 20, 'return_labels': True,
                       'label_path': "",
                       'generate_candidate_centerlines': 6,
                       'compute_all': True}
        # Compute features specified by args
        features = self.compute_features(**input_param)

        # Cast np arrays in feature dict to f32 to save disk space
        features = cast_dict_f32(features, {})        
        item = self.getitem(features, idx)
        model_input = self.collate([item])
        model_input = self.todevice(model_input[0])
        
        return model_input
    
    def getitem(self, data, idx):
        """Returns a single example from dataset
            Args:
                idx: Index of example
            Returns:
                output: Necessary values for example
        """
        example = {}
        example['idx'] = idx

        example['seq_id'] = data['SEQ_ID']
        example['city'] = data['CITY_NAME']
        # Get feature helpers
        if 'TRANSLATION' in data:
            example['translation'] = np.array(data['TRANSLATION'])
        if 'ROTATION' in data:
            example['rotation'] = np.array(data['ROTATION'])

        # Get focal agent features
        example['agent_xy_features'] = data['AGENT']['XY_FEATURES']
        if 'LABELS' in data['AGENT']:
            example['agent_xy_labels'] = data['AGENT']['LABELS']
        else:
            example['agent_xy_labels'] = np.zeros((self.outsteps, 2), dtype=np.float)

        agent_str = '_FULL' if self.is_oracle else '_PARTIAL'

        # Get centerline for IFC
        if not self.is_oracle:
            example['agent_oracle_centerline'] = data['AGENT'][self.heuristic_str+'ORACLE_CENTERLINE_NORMALIZED'+agent_str]
            example['agent_oracle_centerline_lengths'] = example['agent_oracle_centerline'].shape[0]

            # Add noise
            if self.mode == 'train':
                rotation_sign = 1.0 if np.random.binomial(1, 0.5) == 1 else -1.0
                rotation = np.random.random() * 27.0 * rotation_sign
                translation_sign = 1.0 if np.random.binomial(1, 0.5) == 1 else -1.0
                translation = np.random.random(2) * translation_sign
                agent_all_features = np.vstack([example['agent_xy_features'], example['agent_xy_labels']])
                agent_all_features = self.add_noise(agent_all_features, rotation=rotation, translation=translation)
                example['agent_xy_features'] = agent_all_features[:example['agent_xy_features'].shape[0],:]
                example['agent_xy_labels'] = agent_all_features[example['agent_xy_features'].shape[0]:,:]
                example['agent_oracle_centerline'] = self.add_noise(example['agent_oracle_centerline'], rotation=rotation, translation=translation)
        else:

            example['agent_oracle_centerline'] = data['AGENT']['TEST_CANDIDATE_CENTERLINE_NORMALIZED'+agent_str]
            example['agent_oracle_centerline_lengths'] = [x.shape[0] for x in example['agent_oracle_centerline']]

            # Add noise
            if self.mode == 'train':
                rotation_sign = 1.0 if np.random.binomial(1, 0.5) == 1 else -1.0
                rotation = np.random.random() * 27.0 * rotation_sign
                translation_sign = 1.0 if np.random.binomial(1, 0.5) == 1 else -1.0
                translation = np.random.random(2) * translation_sign
                agent_all_features = np.vstack([example['agent_xy_features'], example['agent_xy_labels']])
                agent_all_features = self.add_noise(agent_all_features, rotation=rotation, translation=translation)
                example['agent_xy_features'] = agent_all_features[:example['agent_xy_features'].shape[0],:]
                example['agent_xy_labels'] = agent_all_features[example['agent_xy_features'].shape[0]:,:]
                example['agent_oracle_centerline'] = [self.add_noise(x, rotation=rotation, translation=translation) for x in example['agent_oracle_centerline']]

        # Compute delta xy coordinates if required
        if self.delta:
            padded_xy_delta, padded_labels_delta, ref_start, ref_end = self.relative_distance_with_labels(example['agent_xy_features'], example['agent_xy_labels'])
            example['agent_xy_features_delta'] = padded_xy_delta
            example['agent_xy_labels_delta'] = padded_labels_delta
            example['agent_xy_ref_start'] = ref_start
            example['agent_xy_ref_end'] = ref_end

        # Get social agent features
        num_social_agents = 0
        if self.social_features_flag:
            social = defaultdict(list)
            for social_num, social_features in enumerate(data['SOCIAL']):
                if social_num >= self.max_social_agents:
                    break
                tstamps = social_features['TSTAMPS']
                # Check if social agent has 2 seconds of history
                if social_features['XY_FEATURES'].shape[0] == self.timesteps:

                    # Compute mask for agents that don't have information for all timesteps
                    mask = np.full(self.timesteps + self.outsteps, False)
                    mask[tstamps] = True
                    input_mask = mask[:self.timesteps]
                    label_mask = mask[self.timesteps:]
                    social['social_input_mask'].append(input_mask)
                    social['social_label_mask'].append(label_mask)

                    # Add noise
                    if self.mode == 'train':
                        if 'LABELS' in social_features and len(social_features['LABELS']) > 0:
                            all_features = np.vstack([social_features['XY_FEATURES'], social_features['LABELS']])
                        else:
                            all_features = social_features['XY_FEATURES']
                        all_features = self.add_noise(all_features, rotation=rotation, translation=translation)
                        social_features['XY_FEATURES'] = all_features[:social_features['XY_FEATURES'].shape[0], :]
                        if 'LABELS' in social_features:
                            social_features['LABELS'] = all_features[social_features['XY_FEATURES'].shape[0]:, :]

                    # Get xy coordinates
                    padded_xy = np.zeros((self.timesteps, 2), dtype=np.float)
                    padded_xy[input_mask] = social_features['XY_FEATURES']
                    social['social_xy_features'].append(padded_xy)

                    # Get labels
                    labels = np.array([])
                    if 'LABELS' in social_features:
                        labels = social_features['LABELS']
                    padded_labels = np.zeros((self.outsteps, 2), dtype=np.float)
                    if len(labels) > 0:
                        padded_labels[label_mask] = labels
                    social['social_xy_labels'].append(padded_labels)

                    if len(labels) == 0 or self.mode != 'train':
                        social_str = "_FULL" if self.is_oracle else "_PARTIAL"
                    else:
                        social_str = self.str
                    # Get centerline for IFC

                    if self.mode == 'train':
                        social_features[self.heuristic_str+'ORACLE_CENTERLINE_NORMALIZED'+social_str] = self.add_noise(social_features[self.heuristic_str+'ORACLE_CENTERLINE_NORMALIZED'+social_str], rotation=rotation, translation=translation)
                    social['social_oracle_centerline'].append(social_features[self.heuristic_str+'ORACLE_CENTERLINE_NORMALIZED'+social_str])
                    social['social_oracle_centerline_lengths'].append(social_features[self.heuristic_str + 'ORACLE_CENTERLINE_NORMALIZED'+social_str].shape[0])              
                    num_social_agents += 1

        # Pad centerlines
        social_max_pad = np.max(social['social_oracle_centerline_lengths'])
        if social_max_pad < np.max(example['agent_oracle_centerline_lengths']):
            social_max_pad = np.max(example['agent_oracle_centerline_lengths'])
        for index, elem in enumerate(social['social_oracle_centerline']):
            num_pad = social_max_pad - elem.shape[0]
            padded_elem = np.pad(elem, ((0, num_pad), (0, 0)), 'constant', constant_values=(0.,))
            social['social_oracle_centerline'][index] = padded_elem

        if self.is_oracle:
            max_pad = social_max_pad
            for index, elem in enumerate(example['agent_oracle_centerline']):
                num_pad = max_pad - elem.shape[0]
                padded_elem = np.pad(elem, ((0, num_pad),(0, 0)), 'constant', constant_values=(0.,))
                example['agent_oracle_centerline'][index] = padded_elem
            example['agent_oracle_centerline'] = np.array(example['agent_oracle_centerline'])
            example['agent_oracle_centerline_lengths'] = np.array(example['agent_oracle_centerline_lengths'])
        else:
            example['agent_oracle_centerline'] = np.pad(example['agent_oracle_centerline'], ((0, social_max_pad - example['agent_oracle_centerline'].shape[0]),(0,0)), 'constant', constant_values=(0.,))

        social = {key: np.array(value) for key, value in social.items()}

        # Compute delta xy coordinates if required
        if self.delta:
            padded_social_xy_delta, padded_social_labels_delta, social_ref_start, social_ref_end = self.relative_distance_with_labels(social['social_xy_features'], social['social_xy_labels'])
            social['social_xy_features_delta'] = padded_social_xy_delta
            social['social_xy_labels_delta'] = padded_social_labels_delta
            social['social_xy_ref_start'] = social_ref_start
            social['social_xy_ref_end'] = social_ref_end

        example.update(social)
        example['num_social_agents'] = num_social_agents

        # Create adjacency matrix
        adjacency = np.zeros((self.timesteps, num_social_agents+1, num_social_agents+1))
        label_adjacency = np.zeros((self.outsteps, num_social_agents+1, num_social_agents+1))

        # Focal agent is always present
        # Remove self loop
        adjacency[:, 0, :] = 1
        label_adjacency[:, 0, :] = 1
        for social_agent, input_mask in enumerate(example['social_input_mask']):
            adjacency[input_mask, social_agent + 1, :] = 1
        for social_agent, input_mask in enumerate(example['social_label_mask']):
            label_adjacency[input_mask, social_agent + 1, :] = 1
        indexer = np.arange(num_social_agents + 1)
        adjacency[:, indexer, indexer] = 0
        label_adjacency[:, indexer, indexer] = 0

        example['adjacency'] = adjacency
        example['label_adjacency'] = label_adjacency

        '''
        get_data_from_batch
        '''
        # Get focal agent features
        agent_features = example['agent_xy_features' + self.delta_str]
        if self.map_features_flag:
            agent_features = torch.cat([agent_features, example['agent_map_features']], dim=-1)
        agent_features = agent_features.astype(np.float32)

        # Get social features
        social_features = example['social_xy_features' + self.delta_str]
        social_label_features = example['social_xy_labels' + self.delta_str]
        social_features = social_features.astype(np.float32)
        social_label_features = social_label_features.astype(np.float32)
        social_input_mask = example['social_input_mask']
        social_label_mask = example['social_label_mask']
        num_agent_mask = np.ones(example['num_social_agents'] + 1, dtype=np.float32)
        # num_agent_mask = (example['num_social_agents'][:, None] >= torch.arange(social_label_mask.size(1) + 1)).astype(np.float32)
        adjacency = example['adjacency'].astype(np.float32)
        label_adjacency = example['label_adjacency'].astype(np.float32)

        # Get labels
        agent_labels = example['agent_xy_labels' + self.delta_str].astype(np.float32)
        social_labels = example['social_xy_labels' + self.delta_str].astype(np.float32)

        # Get IFC features
        ifc_helpers = {}
        ifc_helpers['agent_oracle_centerline'] = example['agent_oracle_centerline'].astype(np.float32)
        ifc_helpers['agent_oracle_centerline_lengths'] = np.int64(example['agent_oracle_centerline_lengths'])
        # ifc_helpers['agent_xy_delta'] = None

        ifc_helpers['social_oracle_centerline'] = example['social_oracle_centerline'].astype(np.float32)
        ifc_helpers['social_oracle_centerline_lengths'] = np.int64(example['social_oracle_centerline_lengths'])
        # ifc_helpers['social_xy_delta'] = None

        ifc_helpers['rotation'] = example['rotation']
        ifc_helpers['translation'] = example['translation']
        ifc_helpers['city'] = example['city']
        ifc_helpers['idx'] = example['seq_id']

        if self.delta:
            ifc_helpers['agent_xy_delta'] = example['agent_xy_ref_end'].astype(np.float32)
            ifc_helpers['social_xy_delta'] = example['social_xy_ref_end'].astype(np.float32)

        input_dict = {'agent_features': agent_features,
                      'ifc_helpers': ifc_helpers,
                      'social_features': social_features,
                      'social_label_features': social_label_features,
                      'adjacency': adjacency,
                      'label_adjacency': label_adjacency,
                      'num_agent_mask': num_agent_mask
                      }

        if self.mode != 'test':
            target_dict = {'agent_labels': agent_labels}
            return input_dict, target_dict
        else:
            return input_dict, None
        
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
    
    def add_noise(self, x, rotation, translation):
        trajectory = LineString(x)
        mat = [1, 0, 0, 1, translation[0], translation[1]]
        trajectory_translated = affine_transform(trajectory, mat)

        # Apply rotation
        trajectory_rotated = np.array(rotate(trajectory_translated, rotation, origin=(0, 0)).coords, dtype=np.float32)
        return trajectory_rotated

    def relative_distance_with_labels(self, input, labels):
        """Compute relative distance from absolute
            Returns:
                reference: First element of the trajectory. Enables going back from relative distance to absolute.
        """
        if len(input.shape) == 3:
            # Change input sequences to relative distances
            input_reference_start = input[:, 0, :]
            input_reference_end = input[:, -1, :]
            input_rel_dist = input - np.pad(input, ((0, 0), (1, 0), (0, 0)), 'constant')[:, :input.shape[1], :]

            # Change output sequences to relative distances
            output_rel_dist = labels - np.concatenate((input[:, -1:, :], labels), axis=1)[:, :labels.shape[1], :]
        else:
            # Change input sequences to relative distances
            input_reference_start = input[0, :]
            input_reference_end = input[-1, :]
            input_rel_dist = input - np.pad(input, ((1, 0), (0, 0)), 'constant')[:input.shape[0], :]

            # Change output sequences to relative distances
            output_rel_dist = labels - np.concatenate((input[-1:,:], labels), axis=0)[:labels.shape[0], :]

        return input_rel_dist, output_rel_dist, input_reference_start, input_reference_end
    
    def collate(self, batch):
        np_str_obj_array_pattern = re.compile(r'[SaUO]')
        error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"
        numpy_type_map = {
            'float64': torch.DoubleTensor,
            'float32': torch.FloatTensor,
            'float16': torch.HalfTensor,
            'int64': torch.LongTensor,
            'int32': torch.IntTensor,
            'int16': torch.ShortTensor,
            'int8': torch.CharTensor,
            'uint8': torch.ByteTensor,
        }

        def pad_batch(batch_dict, max_actors):
            '''
                Pad batch such that all examples have same number of social actors. Allows for batch training of graph models.
            '''
            for key, value in batch_dict.items():
                if key == 'social_oracle_centerline':
                    max_centerline_pad = np.max([x.size(1) for x in value])
                if isinstance(value, dict):
                    batch_dict[key] = pad_batch(value, max_actors)
                elif isinstance(value, list) and isinstance(value[0], torch.Tensor):
                    if 'agent' not in key:
                        for index, elem in enumerate(value):
                            if 'adjacency' not in key:
                                if 'centerline' in key and 'lengths' not in key:
                                    num_centerline_pad = max_centerline_pad - elem.size(1)
                                    if len(elem.size()) == 3:
                                        elem = torch.nn.functional.pad(elem, (0, 0, 0, num_centerline_pad, 0, 0), value=0.)
                                    else:
                                        elem = torch.nn.functional.pad(elem, (0, num_centerline_pad, 0, 0), value=0.)
                                num_pad = max_actors - elem.size(0)
                                if len(elem.size()) == 3:
                                    padded_elem = torch.nn.functional.pad(elem, (0, 0, 0, 0, 0, num_pad))
                                elif len(elem.size()) == 2:
                                    padded_elem = torch.nn.functional.pad(elem, (0, 0, 0, num_pad))
                                else:
                                    padded_elem = torch.nn.functional.pad(elem, (0, num_pad))
                            else:
                                num_pad = max_actors - elem.size(1) + 1
                                padded_elem = torch.nn.functional.pad(elem, (0, num_pad, 0, num_pad, 0, 0))
                            value[index] = padded_elem
                        batch_dict[key] = torch.stack(value)
                    else:
                        try:
                            if ('centerline' in key and 'lengths' not in key) or ('mask' in key):
                                max_pad = np.max([x.size(0) for x in value])
                                for index, elem in enumerate(value):
                                    num_pad = max_pad - elem.size(0)
                                    if len(elem.size()) == 3:
                                        padded_elem = torch.nn.functional.pad(elem, (0, 0, 0, 0, 0, num_pad), value=0.)
                                    elif len(elem.size()) == 1:
                                        padded_elem = torch.nn.functional.pad(elem, (0, num_pad), value=0.)
                                    else:
                                        padded_elem = torch.nn.functional.pad(elem, (0, 0, 0, num_pad), value=0.)
                                    value[index] = padded_elem
                            batch_dict[key] = torch.stack(value)
                        except:
                            if 'centerline' in key and 'lengths' not in key:
                                max_pad = np.max([x.size(1) for x in value])
                                for index, elem in enumerate(value):
                                    num_pad = max_pad - elem.size(1)
                                    if len(elem.size()) == 3:
                                        padded_elem = torch.nn.functional.pad(elem, (0,0,0,num_pad,0,0), value=0.)
                                    else:
                                        padded_elem = torch.nn.functional.pad(elem, (0,0,0,num_pad), value=0.)
                                    value[index] = padded_elem
                            max_actors = 6
                            for index, elem in enumerate(value):
                                num_pad = max_actors - elem.size(0)
                                if len(elem.size()) == 3:
                                    padded_elem = torch.nn.functional.pad(elem, (0,0,0,0,0,num_pad))
                                elif len(elem.size()) == 2:
                                    padded_elem = torch.nn.functional.pad(elem, (0,0,0,num_pad))
                                else:
                                    padded_elem = torch.nn.functional.pad(elem, (0,num_pad))
                                value[index] = padded_elem
                            batch_dict[key] = torch.stack(value)
            return batch_dict

        def collate_batch(batch):
            """Puts each data field into a tensor with outer dimension batch size"""
            elem_type = type(batch[0])
            if isinstance(batch[0], torch.Tensor):
                out = None
                if False:
                    # If we're in a background process, concatenate directly into a
                    # shared memory tensor to avoid an extra copy
                    numel = sum([x.numel() for x in batch])
                    storage = batch[0].storage()._new_shared(numel)
                    out = batch[0].new(storage)
                try:
                    return torch.stack(batch, 0, out=out)
                except:
                    return batch
            elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                    and elem_type.__name__ != 'string_':
                elem = batch[0]
                if elem_type.__name__ == 'ndarray':
                    # array of string classes and object
                    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                        raise TypeError(error_msg_fmt.format(elem.dtype))

                    return collate_batch([torch.from_numpy(b) for b in batch])
                if elem.shape == ():  # scalars
                    py_type = float if elem.dtype.name.startswith('float') else int
                    return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
            elif isinstance(batch[0], float):
                return torch.tensor(batch, dtype=torch.float64)
            elif isinstance(batch[0], int):
                return torch.tensor(batch)
            elif isinstance(batch[0], string_classes):
                return batch
            elif isinstance(batch[0], container_abcs.Mapping):
                return {key: collate_batch([d[key] for d in batch]) for key in batch[0]}
            elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
                return type(batch[0])(*(collate_batch(samples) for samples in zip(*batch)))
            elif isinstance(batch[0], container_abcs.Sequence):
                transposed = zip_longest(*batch)
                return [collate_batch(samples) for samples in transposed]
            else:
                return batch
            raise TypeError((error_msg_fmt.format(type(batch[0]))))

        batch = collate_batch(batch)
        max_actors = np.max([x.shape[0] for x in batch[0]['social_features']])
        batch[0] = pad_batch(batch[0], max_actors)
        batch[1] = pad_batch(batch[1], max_actors)
        return batch
        
    def compute_features(self, path, xy_features_flag=True, xy_features_normalize_flag=True,
                         map_features_flag=True, social_features_flag=True, timesteps=20, avm=None,
                         mfu=None, return_labels=False, label_path="", generate_candidate_centerlines=0,
                         compute_all=False):
        """
        Compute features for the given sequence
            Args:
                path (str): Location of the .csv sequence data file
                xy_features_flag (bool): Use xy coordinate features
                xy_features_normalize_flag (bool): Normalize xy features to constraint start of
                    sequence to be (0,0) and end of sequence to be on the positive x axis
                map_features_flag (bool): Compute map features if true
                social_features_flag (bool): Compute social features if true
                timesteps (int): Number of input timesteps (10 timesteps = 1 second)
                avm (ArgoverseMap object): default None. Pass an object if calling this function in a
                    loop to avoid redundant computation
                return_labels (bool): Compute the labels for the given sequence if true
                label_path (str): Path to separate label data file, if necessary
    
            Returns:
                features (dict): Dictionary of features
                feature_helpers (dict) (Only when map features are computed): Dictionary that stores
                    statistics of transformations for map features
        """
    
        def normalize_xy(xy_locations, translation=None, rotation=None, to_rotate=True):
            """
            Translate and rotate the input data so that the first timestep is (0,0) and the last
            timestep lies on the positive x axis.
                Args:
                    xy_locations (numpy array): XY positions for the trajectory
                Returns:
                    xy_locations_normalized (numpy array): normalized XY positions
                    feature_helpers (dict): Dictionary that stores the rotations and translations
                        applied to the trajectory
            """
            # Apply translation
            if xy_locations.shape[0] > 1:
                trajectory = LineString(xy_locations)
                if translation is None:
                    translation = [-xy_locations[0, 0], -xy_locations[0, 1]]
                mat = [1, 0, 0, 1, translation[0], translation[1]]
                trajectory_translated = affine_transform(trajectory, mat)
    
                # Apply rotation
                if to_rotate:
                    if rotation is None:
                        rotation = -np.degrees(np.arctan2(trajectory_translated.coords[-1][1],
                                               trajectory_translated.coords[-1][0]))
    
                    trajectory_rotated = np.array(rotate(trajectory_translated, rotation,
                                                  origin=(0, 0)).coords)
                    return trajectory_rotated, {'TRANSLATION': translation, 'ROTATION': rotation}
                else:
                    return trajectory_translated, {'TRANSLATION': translation, 'ROTATION': None}
            else:
                if translation is None:
                    return np.zeros_like(xy_locations, dtype=np.float), \
                           {'TRANSLATION': [-xy_locations[0, 0], -xy_locations[0, 1]], 'ROTATION': None}
                else:
                    return np.array([[xy_locations[0, 0]+translation[0],
                                     xy_locations[0, 0]+translation[1]]]), \
                           {'TRANSLATION': translation, 'ROTATION': None}
    
        def compute_xy_features(xy_locations, normalize=True, timesteps=20):
            """
            Compute XY features for the given sequence
                Args:
                    xy_locations (numpy array): XY positions for the track
                    normalize (bool): Normalize xy features to constraint start of sequence to be (0,0)
                        and end of sequence to be on the positive x axis
                    timesteps (int): Timesteps for which feature computation needs to be done
                        (10 timesteps = 1 second)
                Returns:
                    xy_features (numpy array): XY features for the given input positions
                    feature_helpers (dict) (Only when normalize=True): Translation and rotations
                        applied to the input data. This information can be used later to denormalize
                        the features.
            """
            # Apply normalization
            if normalize:
                xy_locations, feature_helpers = normalize_xy(xy_locations)
                return xy_locations, feature_helpers
            return xy_locations, None
    
        def compute_map_features(xy_locations, city, timesteps=20, avm=None, mfu=None, rotation=None,
                                 translation=None, labels=None, generate_candidate_centerlines=0,
                                 compute_all=False):
            """
            Compute map based features for the given sequence
                Args:
                    xy_locations (numpy array): XY positions for the track
                    city (string): Name of the city
                    timesteps (int): Timesteps for which feature computation needs to be done
                        (10 timesteps = 1 second)
                    avm (ArgoverseMap object): default None. Pass an object if calling this function in
                        a loop to avoid redundant computation
                Returns:
                    nt_distances_oracle (numpy array): normal and tangential distances for oracle
                        centerline
                    map_feature_helpers (dict): Dictionary containing helpers for map features
            """
            def remove_repeated(centerlines):
                remove_elements = np.zeros(len(centerlines))
                for x in range(len(centerlines)):
                    for y in range(x+1, len(centerlines)):
                        if centerlines[x].shape == centerlines[y].shape:
                            if remove_elements[y] == 0:
                                if np.all(centerlines[x] == centerlines[y]):
                                    remove_elements[y] = 1
                return np.array(centerlines)[remove_elements == 0]
    
            def additional_centerline_features(centerline, xy_locations, save_str="", heuristic=False):
                """
                Compute additional centerline features like curvature and nearest neighbours to
                xy location
                """
                # Compute nearest point to each xy location on the centerline
                distances = cdist(xy_locations, centerline)
                min_dist = np.argmin(distances, axis=1)
                indexing_array = []
                heuristic_str = 'HEURISTIC_' if heuristic else ''
                #####
                centerline_span = 10
                #####
                for i in range(-centerline_span, centerline_span):
                    loc = min_dist + i
                    min_mask = loc < 0
                    max_mask = loc >= centerline.shape[0]
    
                    loc[min_mask] = 0
                    loc[max_mask] = centerline.shape[0] - 1
                    indexing_array.append(loc)
    
                indexing_array = np.stack(indexing_array, axis=1)
                arange_array = np.arange(indexing_array.shape[0])
                centerline_features = np.repeat(np.expand_dims(centerline, axis=0),
                                                indexing_array.shape[0], axis=0)[arange_array[:, None],
                                                                                 indexing_array, :]
    
                # Compute angle of change
                velocity_vector = (centerline_features - np.pad(centerline_features, ((0, 0), (1, 0), (0, 0)), 'constant')[:, :centerline_features.shape[1], :])[:, 1:, :] # NOQA
                absolute_angles = np.degrees(np.arctan2(velocity_vector[:, :, 1],
                                             velocity_vector[:, :, 0]))
                relative_angles = (absolute_angles - np.pad(absolute_angles, ((0, 0), (1, 0)), 'constant')[:, :absolute_angles.shape[1]])[:, 1:] # NOQA
    
                # Compute angle between centerline and heading
                heading_vector = (xy_locations - np.pad(xy_locations, ((1, 0), (0, 0)), 'constant')[:xy_locations.shape[0], :]) # NOQA
                heading_angle = np.degrees(np.arctan2(heading_vector[:, 1], heading_vector[:, 0]))
                relative_heading_angle = np.abs(absolute_angles - heading_angle[:, None])
    
                features = {
                        heuristic_str + 'NEAREST_CENTERLINE_FEATURES' + save_str: centerline_features,
                        heuristic_str + 'NEAREST_VELOCITY_VECTOR' + save_str: velocity_vector,
                        heuristic_str + 'NEAREST_ABSOLUTE_ANGLE' + save_str: absolute_angles,
                        heuristic_str + 'NEAREST_RELATIVE_ANGLE' + save_str: relative_angles,
                        heuristic_str + 'NEAREST_RELATIVE_HEADING_ANGLE' + save_str: relative_heading_angle, # NOQA
                        heuristic_str + 'NEAREST_HEADING_ANGLE' + save_str: heading_angle}
    
                # Compute ego centric features
                #####
                ego_features = False
                if ego_features:
                    translation = -xy_locations
                    rotation = -np.degrees(np.arctan2(heading_vector[:, 1], heading_vector[:, 0]))
                    for key in [heuristic_str + 'NEAREST_CENTERLINE_FEATURES' + save_str]:
                        ego_feature = []
                        for index, point in enumerate(xy_locations):
                            ego_feature.append(normalize_xy(features[key][index],
                                               translation=translation[index],
                                               rotation=rotation[index])[0])
                        features[heuristic_str + 'EGO_'+key] = np.array(ego_feature)
                return features
    
            def map_features_helper(locations, dfs_threshold_multiplier=2.0, save_str="", avm=None,
                                    mfu=None, rotation=None, translation=None,
                                    generate_candidate_centerlines=0, compute_all=False):
                # Initialize map utilities if not provided
                if avm is None:
                    avm = ArgoverseMap()
                if mfu is None:
                    mfu = MapFeaturesUtils()
    
                # Get best-fitting (oracle) centerline for current vehicle
                heuristic_oracle_centerline = mfu.get_candidate_centerlines_for_trajectory(self.orig, self.rot, self.attack_params, locations, city, avm=avm, viz=False, max_candidates=generate_candidate_centerlines, mode='train')[0] # NOQA
                features = {
                    "HEURISTIC_ORACLE_CENTERLINE" + save_str: heuristic_oracle_centerline,
                    "HEURISTIC_ORACLE_CENTERLINE_NORMALIZED" + save_str: normalize_xy(heuristic_oracle_centerline, translation=translation, rotation=rotation)[0] # NOQA
                }
    
                # Get top-fitting candidate centerlines for current vehicle (can beused at test time)
                if compute_all:
                    if generate_candidate_centerlines > 0:
                        test_candidate_centerlines = mfu.get_candidate_centerlines_for_trajectory(self.orig, self.rot, self.attack_params, locations, city, avm=avm, viz=False, max_candidates=generate_candidate_centerlines, mode='test') # NOQA
                        features["TEST_CANDIDATE_CENTERLINES" + save_str] = test_candidate_centerlines
    
                    # Apply rotation and translation normalization if specified
                    if rotation is not None or translation is not None:
                        if generate_candidate_centerlines > 0:
                            features['TEST_CANDIDATE_CENTERLINE_NORMALIZED' + save_str] = [normalize_xy(test_candidate_centerline, translation=translation, rotation=rotation)[0] for test_candidate_centerline in test_candidate_centerlines] # NOQA
                return features
    
            map_features = {}
    
            # Compute polyline-based map features considering only first 2 seconds
            xy_partial_trajectory = xy_locations[:timesteps, :]
            map_partial_features = map_features_helper(xy_partial_trajectory, save_str="_PARTIAL",
                                                       avm=avm, rotation=rotation,
                                                       translation=translation,
                                                       generate_candidate_centerlines=generate_candidate_centerlines, # NOQA
                                                       compute_all=compute_all)
            map_features.update(map_partial_features)
    
            # Compute polyline-based map features considering 5 seconds
            if labels is not None:
                xy_full_trajectory = np.concatenate([xy_locations[:timesteps, :], labels], axis=0)
                map_full_features = map_features_helper(xy_full_trajectory, save_str="_FULL", avm=avm,
                                                        rotation=rotation, translation=translation,
                                                        generate_candidate_centerlines=generate_candidate_centerlines, # NOQA
                                                        compute_all=compute_all)
                map_features.update(map_full_features)
    
            # Compute extra map features if specified
            #####
            extra_map_features = True
            if extra_map_features:
                rotated_and_translated_partial_trajectory = normalize_xy(xy_partial_trajectory,
                                                                         translation=translation,
                                                                         rotation=rotation)[0]
                if labels is not None:
                    if len(labels) > 1:
                        rotated_and_translated_label = normalize_xy(labels, translation=translation,
                                                                    rotation=rotation)[0]
                    else:
                        rotated_and_translated_label = normalize_xy(labels, translation=translation,
                                                                    rotation=None, to_rotate=False)[0]
    
                heuristic_extra_features_partial = additional_centerline_features(map_features["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_PARTIAL"], rotated_and_translated_partial_trajectory, save_str="_PARTIAL", heuristic=True) # NOQA
                map_features.update(heuristic_extra_features_partial)
                if labels is not None:
                    heuristic_extra_features_full = additional_centerline_features(map_features["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_FULL"], rotated_and_translated_partial_trajectory, save_str="_FULL", heuristic=True) # NOQA
                    heuristic_extra_features_label_full = additional_centerline_features(map_features["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_FULL"], rotated_and_translated_label, save_str="_LABEL_FULL", heuristic=True) # NOQA
                    heuristic_extra_features_label_partial = additional_centerline_features(map_features["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_PARTIAL"], rotated_and_translated_label, save_str="_LABEL_PARTIAL", heuristic=True) # NOQA
                    map_features.update(heuristic_extra_features_full)
                    map_features.update(heuristic_extra_features_label_full)
                    map_features.update(heuristic_extra_features_label_partial)
    
                if compute_all:
                    if generate_candidate_centerlines > 0:
                        test_extra_features_partial = [additional_centerline_features(test_candidate_centerline_normalized, rotated_and_translated_partial_trajectory, save_str="_PARTIAL") for test_candidate_centerline_normalized in map_features['TEST_CANDIDATE_CENTERLINE_NORMALIZED_PARTIAL']] # NOQA
                        map_features['TEST_CANDIDATE_CENTERLINE_ADDITIONAL_PARTIAL'] = test_extra_features_partial # NOQA
                        if labels is not None:
                            test_extra_features_partial = [additional_centerline_features(test_candidate_centerline_normalized, rotated_and_translated_partial_trajectory, save_str="_FULL") for test_candidate_centerline_normalized in map_features['TEST_CANDIDATE_CENTERLINE_NORMALIZED_FULL']] # NOQA
                            map_features['TEST_CANDIDATE_CENTERLINE_ADDITIONAL_FULL'] = test_extra_features_partial # NOQA
                            extra_features_label_full = additional_centerline_features(map_features["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_FULL"], rotated_and_translated_label, save_str="_LABEL_FULL") # NOQA
                            extra_features_label_partial = additional_centerline_features(map_features["HEURISTIC_ORACLE_CENTERLINE_NORMALIZED_PARTIAL"], rotated_and_translated_label, save_str="_LABEL_PARTIAL") # NOQA
                            map_features.update(extra_features_label_full)
                            map_features.update(extra_features_label_partial)
            return map_features
    
        def compute_social_features(data, city, timesteps=20, avm=None, mfu=None, rotation=None,
                                    translation=None, generate_candidate_centerlines=0):
            social_agents = data[data["OBJECT_TYPE"] != "AGENT"]
            social_features_all = []
            tmap = {tstamp: num for num, tstamp in enumerate(data['TIMESTAMP'].unique())}
            for track_id in social_agents['TRACK_ID'].unique():
                social_features = OrderedDict([])
                social_agent = social_agents[social_agents['TRACK_ID'] == track_id]
                xy_locations = social_agent[['X', 'Y']].values
                ##### apply the transform function
                xy_locations = apply_transform(xy_locations, self.orig, self.rot, self.attack_params)
                #####
                tstamps = np.array([tmap[t] for t in social_agent['TIMESTAMP'].values])
    
                # Remove actors that appear after first 2 seconds
                if tstamps[0] < timesteps:
                    # Remove trajectories that are too small
                    tsteps = np.sum(tstamps < timesteps)
                    if tsteps > 3:
                        labels = xy_locations[tsteps:, :]
                        if len(labels) == 0:
                            labels = None
                        social_features['TSTAMPS'] = tstamps
                        social_features['LABELS_UNNORMALIZED'] = labels
    
                        # Compute XY Features
                        if xy_features_flag:
                            xy_features = xy_locations[:tsteps, :]
                            if xy_features_normalize_flag:
                                xy_features, _ = normalize_xy(xy_locations=xy_features,
                                                              translation=translation,
                                                              rotation=rotation)
                                if labels is not None:
                                    if len(labels) > 1:
                                        social_features["LABELS"] = normalize_xy(xy_locations=labels,
                                                                                 translation=translation, # NOQA
                                                                                 rotation=rotation)[0]
                                    else:
                                        social_features["LABELS"] = normalize_xy(xy_locations=labels,
                                                                                 translation=translation, # NOQA
                                                                                 rotation=None,
                                                                                 to_rotate=False)[0]
                                else:
                                    social_features["LABELS"] = np.array([])
                            social_features['XY_FEATURES'] = xy_features
    
                        # Compute Map Features
                        if map_features_flag:
                            if xy_features_normalize_flag:
                                map_features = compute_map_features(xy_locations=xy_locations,
                                                                    city=city, timesteps=tsteps,
                                                                    avm=avm, mfu=mfu, rotation=rotation,
                                                                    translation=translation,
                                                                    labels=labels,
                                                                    generate_candidate_centerlines=generate_candidate_centerlines, # NOQA
                                                                    compute_all=False)
                            else:
                                map_features = compute_map_features(xy_locations=xy_locations,
                                                                    city=city, timesteps=tsteps,
                                                                    avm=avm, mfu=mfu, labels=labels,
                                                                    generate_candidate_centerlines=generate_candidate_centerlines, # NOQA
                                                                    compute_all=False)
                            social_features.update(map_features)
                        social_features_all.append(social_features)
            return social_features_all
    
        '''
        Compute features for current data sequence
        '''
        path = str(path)
        data = pd.read_csv(path, dtype={"TIMESTAMP": str})
        final_features = OrderedDict([])
    
        seq_id = path.split('/')[-1].split('.')[0]
        final_features['PATH'] = os.path.abspath(path)
        final_features['CITY_NAME'] = data['CITY_NAME'].values[0]
        final_features['SEQ_ID'] = seq_id
    
        # Get focal agent track
        agent_track = data[data["OBJECT_TYPE"] == "AGENT"]
        xy_locations = agent_track[['X', 'Y']].values
        
        #####
        xy_locations[:timesteps, :] = np.matmul(self.rot, (xy_locations[:timesteps, :] - self.orig.reshape(-1, 2)).T).T
        xy_locations[:timesteps, :] = self.correct_speed(xy_locations[:timesteps, :])
        xy_locations[:timesteps, :] = np.matmul(self.rot.T, xy_locations[:timesteps, :].T).T + self.orig.reshape(-1, 2)
        #print("location before transform :", xy_locations)
        xy_locations = apply_transform(xy_locations, self.orig, self.rot, self.attack_params)
        #print("location after transform :", xy_locations)
        #####
        
        agent_features = {}
    
        # Get labels
        labels = None
        if return_labels:
            if label_path == "":
                labels = xy_locations[timesteps:, :]
            else:
                label_data = pd.read_csv(label_path, dtype={"TIMESTAMP": str})
                label_agent_track = label_data[label_data["OBJECT_TYPE"] == "AGENT"]
                labels = label_agent_track[['X', 'Y']].values
                final_features["LABELS_PATH"] = os.path.abspath(label_path)
            agent_features['LABELS_UNNORMALIZED'] = labels
    
        # Get XY input features
        if xy_features_flag:
            xy_features, xy_feature_helpers = compute_xy_features(xy_locations=xy_locations[:timesteps, :], # NOQA
                                                                  normalize=xy_features_normalize_flag,
                                                                  timesteps=timesteps)
            agent_features['XY_FEATURES'] = xy_features
            if xy_feature_helpers is not None:
                final_features.update(xy_feature_helpers)
    
        # Compute map features
        if map_features_flag:
            if xy_features_normalize_flag:
                map_features = compute_map_features(xy_locations=xy_locations,
                                                    city=final_features['CITY_NAME'],
                                                    timesteps=timesteps, avm=avm, mfu=mfu,
                                                    rotation=xy_feature_helpers['ROTATION'],
                                                    translation=xy_feature_helpers['TRANSLATION'],
                                                    labels=labels,
                                                    generate_candidate_centerlines=generate_candidate_centerlines, # NOQA
                                                    compute_all=compute_all)
            else:
                map_features = compute_map_features(xy_locations=xy_locations,
                                                    city=final_features['CITY_NAME'],
                                                    timesteps=timesteps, avm=avm, mfu=mfu,
                                                    labels=labels,
                                                    generate_candidate_centerlines=generate_candidate_centerlines, # NOQA
                                                    compute_all=compute_all)
            agent_features.update(map_features)
    
        # Compute social features
        if social_features_flag:
            if xy_features_normalize_flag:
                social_features = compute_social_features(data=data, city=final_features['CITY_NAME'],
                                                          timesteps=timesteps, avm=avm, mfu=mfu,
                                                          rotation=xy_feature_helpers['ROTATION'],
                                                          translation=xy_feature_helpers['TRANSLATION'],
                                                          generate_candidate_centerlines=generate_candidate_centerlines) # NOQA
            else:
                social_features = compute_social_features(data=data, city=final_features['CITY_NAME'],
                                                          timesteps=timesteps, avm=avm, mfu=mfu,
                                                          generate_candidate_centerlines=generate_candidate_centerlines) # NOQA
            final_features['SOCIAL'] = social_features
    
        # Compute Labels
        if return_labels:
            final_features["LABELS"] = normalize_xy(xy_locations=labels,
                                                    translation=final_features['TRANSLATION'],
                                                    rotation=final_features['ROTATION'])[0]
            agent_features['LABELS'] = final_features['LABELS']
    
        if bool(agent_features):
            final_features['AGENT'] = agent_features
        return final_features
    
    def correct_speed(self, history):   # inputs history points in the agents coordinate system and corrects its speed
        # calculating the minimum r of the attacking turn
        border1 = self.attack_params["smooth-turn"]["border"]
        border2 = self.attack_params["double-turn"]["border"]
        border3 = self.attack_params["ripple-road"]["border"]
        l_range = min(border1, border2, border3)
        r_range = max(border1 + 10, border2 + 20 + self.attack_params["double-turn"]["l"], border3 + self.attack_params["ripple-road"]["l"])

        search_points = np.linspace(l_range, r_range, 100)
        search_point_rs = self.calc_radius(search_points)
        min_r = search_point_rs.min()
        g = 9.8
        miu_s = 0.7
        max_speed = np.sqrt(miu_s * g * min_r)
        self.max_speed = max_speed
        current_speed = np.sqrt(((history[-1] - history[-2])**2).sum()) * 10
        self.current_speed = current_speed
        if current_speed <= max_speed:
            return history
        self.scale_factor = max_speed / current_speed
        return history * self.scale_factor

    def calc_curvature(self, x):
        numerator = self.attack_function.f_zegond(x)
        denominator = (1 + self.attack_function.f_prime(x)**2)**1.5
        return numerator / denominator

    def calc_radius(self, x):
        curv = self.calc_curvature(x)
        ret = np.zeros_like(x)
        ret[curv == 0] = 1000_000_000_000
        ret[curv != 0] = 1 / np.abs(curv[curv != 0])
        return ret
    
    def todevice(self, inp):
        if(type(inp).__name__ == 'Tensor'):
            inp = inp.to(self.device)
        elif(type(inp).__name__ == 'dict'):
            for key in inp:
                inp[key] = self.todevice(inp[key])
            return inp
        return inp
        
def cast_dict_f32(old_dict, new_dict):
    """
    Returns a copy of old_dict, with all np arrays cast to float32 in order to save disk space
    """
    for key, item in old_dict.items():
        if '_FULL' in key:
            continue
        if isinstance(item, dict):
            new_dict[key] = cast_dict_f32(item, {})
        elif isinstance(item, list) and isinstance(item[0], dict):
            new_dict[key] = [cast_dict_f32(x, {}) for x in item]
        else:
            if isinstance(item, (np.ndarray, np.int64, np.float64)):
                if item.dtype == np.int64:
                    new_dict[key] = item.astype(np.int32)
                elif item.dtype == np.float64:
                    new_dict[key] = item.astype(np.float32)
            elif isinstance(item, list) and isinstance(item[0], (np.ndarray, np.int64, np.float64)):
                new_dict[key] = [x.astype(np.float32) for x in item]
            else:
                new_dict[key] = item
    return new_dict        

class MapFeaturesUtils:
    """Utils for computation of map-based features."""
    def __init__(self):
        """Initialize class."""
        self._MANHATTAN_THRESHOLD = 5.0
        self._DFS_THRESHOLD_FRONT_SCALE = 45.0
        self._DFS_THRESHOLD_BACK_SCALE = 40.0
        self._MAX_SEARCH_RADIUS_CENTERLINES = 50.0
        self._MAX_CENTERLINE_CANDIDATES_TEST = 6

    def get_point_in_polygon_score(self, lane_seq: List[int],
                                   xy_seq: np.ndarray, city_name: str,
                                   avm: ArgoverseMap) -> int:
        """Get the number of coordinates that lie insde the lane seq polygon.

        Args:
            lane_seq: Sequence of lane ids
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            avm: Argoverse map_api instance
        Returns:
            point_in_polygon_score: Number of coordinates in the trajectory that lie within the
            lane sequence
        """
        lane_seq_polygon = cascaded_union([
            Polygon(avm.get_lane_segment_polygon(lane, city_name)).buffer(0)
            for lane in lane_seq
        ])
        point_in_polygon_score = 0
        for xy in xy_seq:
            point_in_polygon_score += lane_seq_polygon.contains(Point(xy))
        return point_in_polygon_score

    def sort_lanes_based_on_point_in_polygon_score(
            self,
            lane_seqs: List[List[int]],
            xy_seq: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
    ) -> List[List[int]]:
        """Filter lane_seqs based on the number of coordinates inside the bounding polygon of lanes.

        Args:
            lane_seqs: Sequence of lane sequences
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            avm: Argoverse map_api instance
        Returns:
            sorted_lane_seqs: Sequences of lane sequences sorted based on the point_in_polygon score

        """
        point_in_polygon_scores = []
        for lane_seq in lane_seqs:
            point_in_polygon_scores.append(
                self.get_point_in_polygon_score(lane_seq, xy_seq, city_name,
                                                avm))
        randomized_tiebreaker = np.random.random(len(point_in_polygon_scores))
        sorted_point_in_polygon_scores_idx = np.lexsort(
            (randomized_tiebreaker, np.array(point_in_polygon_scores)))[::-1]
        sorted_lane_seqs = [
            lane_seqs[i] for i in sorted_point_in_polygon_scores_idx
        ]
        sorted_scores = [
            point_in_polygon_scores[i]
            for i in sorted_point_in_polygon_scores_idx
        ]
        return sorted_lane_seqs, sorted_scores

    def get_heuristic_centerlines_for_test_set(
            self,
            lane_seqs: List[List[int]],
            xy_seq: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
            max_candidates: int,
            scores: List[int],
    ) -> List[np.ndarray]:
        """Sort based on distance along centerline and return the centerlines.

        Args:
            lane_seqs: Sequence of lane sequences
            xy_seq: Trajectory coordinates
            city_name: City name (PITT/MIA)
            avm: Argoverse map_api instance
            max_candidates: Maximum number of centerlines to return
        Return:
            sorted_candidate_centerlines: Centerlines in the order of their score

        """
        aligned_centerlines = []
        diverse_centerlines = []
        diverse_scores = []

        # Get first half as aligned centerlines
        aligned_cl_count = 0
        for i in range(len(lane_seqs)):
            lane_seq = lane_seqs[i]
            score = scores[i]
            diverse = True
            centerline = avm.get_cl_from_lane_seq([lane_seq], city_name)[0]
            if aligned_cl_count < int(max_candidates / 2):
                start_dist = LineString(centerline).project(Point(xy_seq[0]))
                end_dist = LineString(centerline).project(Point(xy_seq[-1]))
                if end_dist > start_dist:
                    aligned_cl_count += 1
                    aligned_centerlines.append(centerline)
                    diverse = False
            if diverse:
                diverse_centerlines.append(centerline)
                diverse_scores.append(score)

        num_diverse_centerlines = min(len(diverse_centerlines),
                                      max_candidates - aligned_cl_count)

        test_centerlines = aligned_centerlines
        if num_diverse_centerlines > 0:
            probabilities = ([
                float(score + 1) / (sum(diverse_scores) + len(diverse_scores))
                for score in diverse_scores
            ] if sum(diverse_scores) > 0 else [1.0 / len(diverse_scores)] *
                             len(diverse_scores))
            diverse_centerlines_idx = np.random.choice(
                range(len(probabilities)),
                num_diverse_centerlines,
                replace=False,
                p=probabilities,
            )
            diverse_centerlines = [
                diverse_centerlines[i] for i in diverse_centerlines_idx
            ]
            test_centerlines += diverse_centerlines

        return test_centerlines

    def get_candidate_centerlines_for_trajectory(
            self,
            orig,
            rot,
            attack_params,
            xy: np.ndarray,
            city_name: str,
            avm: ArgoverseMap,
            viz: bool = False,
            max_search_radius: float = 50.0,
            seq_len: int = 50,
            max_candidates: int = 100,
            mode: str = "test",
    ) -> List[np.ndarray]:
        """Get centerline candidates upto a threshold.

        Algorithm:
        1. Take the lanes in the bubble of last observed coordinate
        2. Extend before and after considering all possible candidates
        3. Get centerlines based on point in polygon score.

        Args:
            xy: Trajectory coordinates, 
            city_name: City name, 
            avm: Argoverse map_api instance, 
            viz: Visualize candidate centerlines, 
            max_search_radius: Max search radius for finding nearby lanes in meters,
            seq_len: Sequence length, 
            max_candidates: Maximum number of centerlines to return, 
            mode: train/val/test mode

        Returns:
            candidate_centerlines: List of candidate centerlines

        """
        ##### return xy points to first position
        xy = apply_transform(xy, orig, rot, attack_params, inverse=True)
        #####
        #print("after reverse : ", xy)
        
        # Get all lane candidates within a bubble
        curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(
            xy[-1, 0], xy[-1, 1], city_name, self._MANHATTAN_THRESHOLD)

        # Keep expanding the bubble until at least 1 lane is found
        while (len(curr_lane_candidates) < 1
               and self._MANHATTAN_THRESHOLD < max_search_radius):
            self._MANHATTAN_THRESHOLD *= 2
            curr_lane_candidates = avm.get_lane_ids_in_xy_bbox(
                xy[-1, 0], xy[-1, 1], city_name, self._MANHATTAN_THRESHOLD)

        assert len(curr_lane_candidates) > 0, "No nearby lanes found!!"

        # Set dfs threshold
        dfs_threshold_front = 150.0
        dfs_threshold_back = 150.0

        # DFS to get all successor and predecessor candidates
        obs_pred_lanes: List[Sequence[int]] = [] # NOQA
        for lane in curr_lane_candidates:
            candidates_future = avm.dfs(lane, city_name, 0,
                                        dfs_threshold_front)
            candidates_past = avm.dfs(lane, city_name, 0, dfs_threshold_back,
                                      True)

            # Merge past and future
            for past_lane_seq in candidates_past:
                for future_lane_seq in candidates_future:
                    assert (
                        past_lane_seq[-1] == future_lane_seq[0]
                    ), "Incorrect DFS for candidate lanes past and future"
                    obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])

        # Removing overlapping lanes
        obs_pred_lanes = remove_overlapping_lane_seq(obs_pred_lanes)

        # Sort lanes based on point in polygon score
        obs_pred_lanes, scores = self.sort_lanes_based_on_point_in_polygon_score(
            obs_pred_lanes, xy, city_name, avm)

        # If the best centerline is not along the direction of travel, re-sort
        if mode == "test":
            candidate_centerlines = self.get_heuristic_centerlines_for_test_set(
                obs_pred_lanes, xy, city_name, avm, max_candidates, scores)
        else:
            candidate_centerlines = avm.get_cl_from_lane_seq(
                [obs_pred_lanes[0]], city_name)

        if viz:
            plt.figure(0, figsize=(8, 7))
            for centerline_coords in candidate_centerlines:
                visualize_centerline(centerline_coords)
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "-",
                color="#d33e4c",
                alpha=1,
                linewidth=3,
                zorder=15,
            )

            final_x = xy[-1, 0]
            final_y = xy[-1, 1]

            plt.plot(
                final_x,
                final_y,
                "o",
                color="#d33e4c",
                alpha=1,
                markersize=10,
                zorder=15,
            )
            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            plt.axis("off")
            plt.title(f"Number of candidates = {len(candidate_centerlines)}")
            plt.show()
            
        ##### apply the transform function
        for i in range(len(candidate_centerlines)):
            candidate_centerlines[i] = apply_transform(candidate_centerlines[i], orig, rot, attack_params)
        xy = apply_transform(xy, orig, rot, attack_params)


        return candidate_centerlines

    def compute_map_features(
            self,
            agent_track: np.ndarray,
            obs_len: int,
            seq_len: int,
            raw_data_format: Dict[str, int],
            mode: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute map based features for the given sequence.

        If the mode is test, oracle_nt_dist will be empty, candidate_nt_dist will be populated.
        If the mode is train/val, oracle_nt_dist will be populated, candidate_nt_dist will be empty.

        Args:
            agent_track : Data for the agent track
            obs_len : Length of observed trajectory
            seq_len : Length of the sequence
            raw_data_format : Format of the sequence
            mode: train/val/test mode

        Returns:
            oracle_nt_dist (numpy array): normal and tangential distances for oracle centerline
                map_feature_helpers (dict): Dictionary containing helpers for map features

        """
        # Get observed 2 secs of the agent
        agent_xy = agent_track[:, [raw_data_format["X"], raw_data_format["Y"]
                                   ]].astype("float")
        agent_track_obs = agent_track[:obs_len]
        agent_xy_obs = agent_track_obs[:, [
            raw_data_format["X"], raw_data_format["Y"]
        ]].astype("float")

        # Get API for Argo Dataset map
        avm = ArgoverseMap()

        city_name = agent_track[0, raw_data_format["CITY_NAME"]]

        # Get candidate centerlines using observed trajectory
        if mode == "test":
            oracle_centerline = np.full((seq_len, 2), None)
            oracle_nt_dist = np.full((seq_len, 2), None)
            candidate_centerlines = self.get_candidate_centerlines_for_trajectory(
                agent_xy_obs,
                city_name,
                avm,
                viz=False,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                max_candidates=self._MAX_CENTERLINE_CANDIDATES_TEST,
            )

            # Get nt distance for the entire trajectory using candidate centerlines
            candidate_nt_distances = []
            for candidate_centerline in candidate_centerlines:
                candidate_nt_distance = np.full((seq_len, 2), None)
                candidate_nt_distance[:obs_len] = get_nt_distance(
                    agent_xy_obs, candidate_centerline)
                candidate_nt_distances.append(candidate_nt_distance)

        else:
            oracle_centerline = self.get_candidate_centerlines_for_trajectory(
                agent_xy,
                city_name,
                avm,
                viz=False,
                max_search_radius=self._MAX_SEARCH_RADIUS_CENTERLINES,
                seq_len=seq_len,
                mode=mode,
            )[0]
            candidate_centerlines = [np.full((seq_len, 2), None)]
            candidate_nt_distances = [np.full((seq_len, 2), None)]

            # Get NT distance for oracle centerline
            oracle_nt_dist = get_nt_distance(agent_xy,
                                             oracle_centerline,
                                             viz=False)

        map_feature_helpers = {
            "ORACLE_CENTERLINE": oracle_centerline,
            "CANDIDATE_CENTERLINES": candidate_centerlines,
            "CANDIDATE_NT_DISTANCES": candidate_nt_distances,
        }

        return oracle_nt_dist, map_feature_helpers


def apply_transform(points, orig, rot, params, inverse = False):
    attack_function = Combination(params)
    points = np.matmul(rot, (points - orig.reshape(-1, 2)).T).T
    if(inverse):
        points[:, 1] -= attack_function.f(points[:, 0])
    else:
        points[:, 1] += attack_function.f(points[:, 0])
    points = np.matmul(rot.T, points.T).T + orig.reshape(-1, 2)
    return points
