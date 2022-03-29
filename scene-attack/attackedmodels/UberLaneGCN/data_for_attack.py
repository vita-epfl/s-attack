import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import os
import copy


def read_argo_data(idx, avl):
    city = copy.deepcopy(avl[idx].city)

    """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME"""
    df = copy.deepcopy(avl[idx].seq_df)

    agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))
    mapping = dict()
    for i, ts in enumerate(agt_ts):
        mapping[ts] = i

    trajs = np.concatenate((
        df.X.to_numpy().reshape(-1, 1),
        df.Y.to_numpy().reshape(-1, 1)), 1)

    steps = [mapping[x] for x in df['TIMESTAMP'].values]
    steps = np.asarray(steps, np.int64)

    objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups
    keys = list(objs.keys())
    obj_type = [x[1] for x in keys]

    agt_idx = obj_type.index('AGENT')
    idcs = objs[keys[agt_idx]]

    agt_traj = trajs[idcs]
    agt_step = steps[idcs]

    del keys[agt_idx]
    ctx_trajs, ctx_steps = [], []
    for key in keys:
        idcs = objs[key]
        ctx_trajs.append(trajs[idcs])
        ctx_steps.append(steps[idcs])

    data = dict()
    data['city'] = city
    data['trajs'] = [agt_traj] + ctx_trajs
    data['steps'] = [agt_step] + ctx_steps
    return data


def get_obj_feats(data):
    orig = data['trajs'][0][19].copy().astype(np.float32)


    pre = data['trajs'][0][18] - orig
    theta = np.pi - np.arctan2(pre[1], pre[0])

    rot = np.asarray([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]], np.float32)

    feats, ctrs, gt_preds, has_preds, step0 = [], [], [], [], []
    for traj, step in zip(data['trajs'], data['steps']):
        if 19 not in step:
            continue

        gt_pred = np.zeros((30, 2), np.float32)
        has_pred = np.zeros(30, np.bool)
        future_mask = np.logical_and(step >= 20, step < 50)
        post_step = step[future_mask] - 20
        post_traj = traj[future_mask]
        gt_pred[post_step] = post_traj
        has_pred[post_step] = 1

        obs_mask = step < 20
        step = step[obs_mask]
        traj = traj[obs_mask]
        idcs = step.argsort()
        step = step[idcs]
        traj = traj[idcs]

        for i in range(len(step)):
            if step[i] == 19 - (len(step) - 1) + i:
                break
        step = step[i:]
        traj = traj[i:]

        feat = np.zeros((20, 3), np.float32)
        feat[step, :2] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
        feat[step, 2] = 1.0

        x_min, x_max, y_min, y_max = [-100.0, 100.0, -100.0, 100.0]
        if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
            continue

        ctrs.append(feat[-1, :2].copy())
        # feat[1:, :2] -= feat[:-1, :2]
        # feat[step[0], :2] = 0
        step0.append(step[0])
        feats.append(feat)
        gt_preds.append(gt_pred)
        has_preds.append(has_pred)

    feats = np.asarray(feats, np.float32)
    ctrs = np.asarray(ctrs, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)
    has_preds = np.asarray(has_preds, np.bool)

    data['feats'] = feats
    data['ctrs'] = ctrs
    data['orig'] = orig
    data['theta'] = theta
    data['rot'] = rot
    data['gt_preds'] = gt_preds
    data['has_preds'] = has_preds
    data['step0'] = step0
    return data


def get_lane_graph(data, am, lanes):

    lane_ids = list(lanes.keys())
    ctrs, feats, turn, control, intersect = [], [], [], [], []
    for lane_id in lane_ids:
        lane = lanes[lane_id]
        ctrln = lane.centerline
        num_segs = len(ctrln) - 1

        ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
        feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

        x = np.zeros((num_segs, 2), np.float32)
        if lane.turn_direction == 'LEFT':
            x[:, 0] = 1
        elif lane.turn_direction == 'RIGHT':
            x[:, 1] = 1
        else:
            pass
        turn.append(x)

        control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
        intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))

    node_idcs = []
    count = 0
    for i, ctr in enumerate(ctrs):
        node_idcs.append(range(count, count + len(ctr)))
        count += len(ctr)
    num_nodes = count

    pre, suc = dict(), dict()
    for key in ['u', 'v']:
        pre[key], suc[key] = [], []
    for i, lane_id in enumerate(lane_ids):
        lane = lanes[lane_id]
        idcs = node_idcs[i]

        pre['u'] += idcs[1:]
        pre['v'] += idcs[:-1]
        if lane.predecessors is not None:
            for nbr_id in lane.predecessors:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    if j < len(node_idcs) and len(node_idcs[j]) > 0:
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])

        suc['u'] += idcs[:-1]
        suc['v'] += idcs[1:]
        if lane.successors is not None:
            for nbr_id in lane.successors:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    if j < len(node_idcs) and len(node_idcs[j]) > 0:
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])

    lane_idcs = []
    for i, idcs in enumerate(node_idcs):
        lane_idcs.append(i * np.ones(len(idcs), np.int64))
    lane_idcs = np.concatenate(lane_idcs, 0)

    pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
    for i, lane_id in enumerate(lane_ids):
        lane = lanes[lane_id]

        nbr_ids = lane.predecessors
        if nbr_ids is not None:
            for nbr_id in nbr_ids:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    pre_pairs.append([i, j])

        nbr_ids = lane.successors
        if nbr_ids is not None:
            for nbr_id in nbr_ids:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    suc_pairs.append([i, j])

        nbr_id = lane.l_neighbor_id
        if nbr_id is not None:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                left_pairs.append([i, j])

        nbr_id = lane.r_neighbor_id
        if nbr_id is not None:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                right_pairs.append([i, j])
    pre_pairs = np.asarray(pre_pairs, np.int64)
    suc_pairs = np.asarray(suc_pairs, np.int64)
    left_pairs = np.asarray(left_pairs, np.int64)
    right_pairs = np.asarray(right_pairs, np.int64)

    graph = dict()
    graph['ctrs'] = np.concatenate(ctrs, 0)
    graph['num_nodes'] = num_nodes
    graph['feats'] = np.concatenate(feats, 0)
    graph['turn'] = np.concatenate(turn, 0)
    graph['control'] = np.concatenate(control, 0)
    graph['intersect'] = np.concatenate(intersect, 0)
    graph['pre'] = [pre]
    graph['suc'] = [suc]
    graph['lane_idcs'] = lane_idcs
    graph['pre_pairs'] = pre_pairs
    graph['suc_pairs'] = suc_pairs
    graph['left_pairs'] = left_pairs
    graph['right_pairs'] = right_pairs

    for k1 in ['pre', 'suc']:
        for k2 in ['u', 'v']:
            graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)

    for key in ['pre', 'suc']:
        graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], 6)
    return graph


def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs