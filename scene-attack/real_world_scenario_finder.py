import osmium as o
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import utm
import time
import pickle
from tqdm import tqdm
from sortedcontainers import SortedSet
from sklearn.cluster import KMeans
import torch
import warnings
from config import get_NewMexico_way_node_locations, get_HongKong_way_node_locations, get_Paris_way_node_locations, get_NewYork_way_node_locations, get_K_Tree_load_dir
from attackedmodels.UberLaneGCN import data_for_attack
from attackedmodels.UberLaneGCN import lanegcn
from attackedmodels.UberLaneGCN.utils import load_pretrain, gpu
from attackedmodels.UberLaneGCN.preprocess_data import to_long, preprocess
from attackedmodels.UberLaneGCN.data import ref_copy, from_numpy
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='debugger arguments')
parser.add_argument("--load_saved_tree", action="store_true")
args = parser.parse_args()
road_type_widths = {
                      'motorway': 5 * 3,
                      'trunk': 5 * 3,
                      'primary': 4.5 * 3,
                      'secondary': 4 * 3,
                      'tertiary': 3.5 * 3,
                      'residential': 3 * 3,
                      'service': 2 * 3,
                      'unclassified': 2 * 3,
                      'pedestrian': 2 * 3,
                      'footway': 1 * 3,
                  }


class HighwayNodeSaverHandler(o.SimpleHandler):
    def __init__(self):
        super(HighwayNodeSaverHandler, self).__init__()
        self.way_node_ids = []
        self.way_types = []
        self.cnt = 0

    def way(self, w):
        if 'highway' in w.tags and w.tags['highway'] in road_type_widths.keys():
            self.cnt += 1
            nodes = []
            for n in w.nodes:
                nodes.append(n.ref)
            if len(nodes) <= 1:
                self.cnt -= 1
                return
            self.way_node_ids.append(nodes)
            self.way_types.append(w.tags['highway'])
            if self.cnt % 500 == 0:
                print("{} ways' nodes has been extracted".format(self.cnt))


class NodeLocationFinderHandler(o.SimpleHandler):
    def __init__(self):
        super(NodeLocationFinderHandler, self).__init__()
        self.node_locations = {}
        self.cnt = 0

    def node(self, n):
        self.cnt += 1
        if self.cnt % 1000 == 0:
            print("{} nodes' latlon have been saved".format(self.cnt))
        self.node_locations[n.id] = [n.location.lat, n.location.lon]


def extract_roads(osmfile):
    """
    inputs a osmfile as input and outputs the ways in the map inside this osm file
    """
    h = HighwayNodeSaverHandler()
    h.apply_file(osmfile)
    print("number of ways:", h.cnt)
    way_node_ids = h.way_node_ids
    way_types = h.way_types
    node_ids_flatten = []
    for way_node_id in way_node_ids:
        node_ids_flatten += way_node_id
    print("number of nodes:", len(node_ids_flatten))
    l = NodeLocationFinderHandler()
    l.apply_file(osmfile)
    node_latlons = np.array(list(l.node_locations.values()))
    tic = time.time()
    x, y, _, _ = utm.from_latlon(node_latlons[:, 0], node_latlons[:, 1])
    print("{} seconds took changing latlon of nodes to x,y coordinates".format(int(time.time() - tic)))
    node_locations = np.zeros_like(node_latlons)
    node_locations[:, 0] = x
    node_locations[:, 1] = y
    node_locations_dict = {}
    for i, id in enumerate(list(l.node_locations.keys())):
        node_locations_dict[id] = node_locations[i]
    way_node_locations = []
    for way_node_id in way_node_ids:
        way_node_location = []
        for id in way_node_id:
            way_node_location.append(node_locations_dict[id])
        way_node_locations.append(np.array(way_node_location))

    return way_node_locations, way_types


def visualize_roads(way_node_locations, min_x, min_y, max_x, max_y, ax):
    ax.set(xlim=(0, max_x - min_x), ylim=(0, max_y - min_y))
    ax.set_aspect('equal', adjustable='box')
    for way_node_location in way_node_locations:
        color = "#5A5A5B"
        way_node_location = np.array(way_node_location)
        ax.plot(
            way_node_location[:, 0] - min_x,
            way_node_location[:, 1] - min_y,
            "-",
            color=color,
            alpha=1,
            linewidth=1,
            zorder=0,
        )


def get_points_in_roads(way_node_locations):
    points = way_node_locations[0]
    for i in range(1, len(way_node_locations)):
        points = np.concatenate([points, way_node_locations[i]], 0)

    way_ids = {}
    for x, y in points:
        way_ids[(x, y)] = []

    for i, node_locations in enumerate(way_node_locations):
        for x, y in node_locations:
            way_ids[(x, y)].append(i)

    return points, way_ids


class Lane:
    def __init__(self, lane_id, ctrs, relations):

        ctrs = np.array(ctrs)
        self.centerline = ctrs
        self.polygon = self.way_to_polygon(ctrs)
        self.turn_direction = self.find_turn_direction(ctrs)
        self.has_traffic_control = self.find_has_traffic_control(ctrs)
        self.is_intersection = self.find_is_intersection()
        self.l_neighbor_id = self.find_l_neighbor_id()
        self.r_neighbor_id = self.find_r_neighbor_id()
        self.predecessors = self.find_predecessors(lane_id, relations)
        self.successors = self.find_successors(lane_id, relations)

    def way_to_polygon(self, centerline):
        # lane_polygon = centerline_to_polygon(centerline[:, :2])
        # return self.append_height_to_2d_city_pt_cloud(lane_polygon, city_name)
        return None

    def find_turn_direction(self, ctrs):
        if len(ctrs) < 3:
            return "NONE"
        x_t = np.gradient(ctrs[:, 0])
        y_t = np.gradient(ctrs[:, 1])

        vel = np.array([[x_t[i], y_t[i]] for i in range(x_t.size)])
        speed = np.sqrt(x_t * x_t + y_t * y_t)
        tangent = np.array([1 / speed] * 2).transpose() * vel
        ss_t = np.gradient(speed)
        xx_t = np.gradient(x_t)
        yy_t = np.gradient(y_t)

        curvature_val = (xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t) ** 1.5
        max_idx = np.argmax(np.abs(curvature_val))
        max_curv = curvature_val[max_idx]
        if (max_curv > 0.1):
            return 'RIGHT'
        elif (max_curv < -0.1):
            return 'LEFT'
        else:
            return 'NONE'

    def find_has_traffic_control(self, ctrs):
        return False

    def find_is_intersection(self):
        return False

    def find_l_neighbor_id(self):
        return None

    def find_r_neighbor_id(self):
        return None

    def find_predecessors(self, lane_id, relations):
        out = np.where(relations[:, lane_id] == 1)
        if (len(out[0]) > 0):
            return out[0]
        else:
            return None

    def find_successors(self, lane_id, relations):
        out = np.where(relations[lane_id] == 1)
        if (len(out[0]) > 0):
            return out[0]
        else:
            return None


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash(self.y)

    def __lt__(self, other):
        return self.y < other.y

    def __le__(self, other):
        return self.y <= other.y

    def __gt__(self, other):
        return self.y > other.y

    def __ge__(self, other):
        return self.y >= other.y

    def __eq__(self, other):
        return self.y == other.y


def get_sample_rectangles(min_x, min_y, max_x, max_y, x_size, y_size, num):
    lower_right_points = np.zeros((num, 2))
    lower_right_points[:, 0] = np.random.uniform(min_x + x_size, max_x, num)
    lower_right_points[:, 1] = np.random.uniform(min_y + y_size, max_y, num)
    return lower_right_points


def get_samples(way_node_locations, min_x, min_y, max_x, max_y, x_size, y_size, num, visualize=False):
    larger_num = 2*num
    nodes, way_ids = get_points_in_roads(way_node_locations)
    nodes = nodes[nodes[:, 0].argsort()]
    points = []
    print(len(nodes))
    for x,y in nodes:
        points.append(Point(x, y))

    rect_ur_coords = get_sample_rectangles(min_x, min_y, max_x, max_y, x_size, y_size, larger_num)
    rect_ur_coords = rect_ur_coords[rect_ur_coords[:, 0].argsort()]

    sample_ways = []
    l_point = r_point = 0
    ys_set = SortedSet()
    is_empty = np.zeros(larger_num)
    index = 0
    for x, y in rect_ur_coords:
        sample_ways.append([])
        # update set of ys
        while r_point < len(nodes) and nodes[r_point, 0] < x:
            # print("adding point {} with coordinates (x,y) = ({},{}) with x {}".format(r_point, nodes[r_point, 0], nodes[r_point, 1], x))
            ys_set.add(points[r_point])
            r_point += 1
        while l_point < len(nodes) and nodes[l_point, 0] < x - x_size:
            # print("deleting point {} with coordinates (x,y) = ({},{})".format(l_point, nodes[l_point, 0], nodes[l_point, 1]))
            ys_set.discard(points[l_point])
            l_point += 1
        # now y of all the points with xs between x-x_size and x are in ys_set
        this_way_ids = set()
        for point in ys_set.irange(Point(0, y - y_size), Point(0, y)):
            for way_id in way_ids[(point.x, point.y)]:
                this_way_ids.add(way_id)
        for way_id in this_way_ids:
            adding_way_node_locations = []
            for n_x, n_y in way_node_locations[way_id]:
                if x - x_size <= n_x <= x and y - y_size <= n_y <= y:
                    adding_way_node_locations.append([n_x, n_y])
            if len(adding_way_node_locations) >= 2:
                sample_ways[-1].append(np.array(adding_way_node_locations))
        if len(sample_ways[-1]) == 0:
            is_empty[index] = 1
        index += 1
    sample_ways = (np.array(sample_ways)[is_empty == 0]).tolist()
    rect_ur_coords = rect_ur_coords[is_empty == 0, :]
    if len(rect_ur_coords) < num:
        print("need {} new samples".format(len(rect_ur_coords)))
        additional_sample_ways, additional_rects = get_samples(way_node_locations, min_x, min_y, max_x, max_y, x_size, y_size, num - len(rect_ur_coords), visualize)
        sample_ways = sample_ways + additional_sample_ways
        rect_ur_coords = np.concatenate([rect_ur_coords, additional_rects], 0)
    sample_ways = sample_ways[:num]
    rect_ur_coords = rect_ur_coords[:num]
    if visualize:
        fig, ax = plt.subplots()
        visualize_roads(way_node_locations, min_x, min_y, max_x, max_y, ax)
        for x, y in rect_ur_coords:
            rect = patches.Rectangle((x - x_size - min_x, y - y_size - min_y), x_size, y_size, linewidth=1,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()
        for i in range(num):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.set_title("Image cropped")
            visualize_roads(way_node_locations, rect_ur_coords[i, 0] - x_size, rect_ur_coords[i, 1] - y_size,
                            rect_ur_coords[i, 0], rect_ur_coords[i, 1], ax1)
            ax2.set_title("Ways selected")
            visualize_roads(sample_ways[i], rect_ur_coords[i, 0] - x_size, rect_ur_coords[i, 1] - y_size,
                            rect_ur_coords[i, 0], rect_ur_coords[i, 1], ax2)
            plt.show()
    return sample_ways, rect_ur_coords


def get_ubers_graph(way_node_locations):
    way_node_locations = np.array(way_node_locations)

    ways_relation = np.zeros((len(way_node_locations), len(way_node_locations)))
    for i in range(len(way_node_locations)):
        for j in range(len(way_node_locations)):
            first = way_node_locations[i]
            second = way_node_locations[j]
            if (np.all(np.around(first[-1], decimals=0) == np.around(second[0], decimals=0))):
                ways_relation[i][j] = 1

    lanes = {}
    for i, way_node_location in enumerate(way_node_locations):
        lane = Lane(i, way_node_location, ways_relation)
        lanes[i] = lane

    for lane_id in list(lanes.keys()):
        lane = lanes[lane_id]
    gr = data_for_attack.get_lane_graph(None, None, lanes)
    # now the real pre processing begins (only pre processing the graph)
    graph = dict()
    for key in ['lane_idcs', 'ctrs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'feats']:
        graph[key] = ref_copy(gr[key])
    graph['idx'] = 0
    out = preprocess(to_long(gpu(dict(from_numpy(graph)))), 6)
    gr['left'] = out['left']
    gr['right'] = out['right']

    return gr


def get_ubers_node_features(sample_ways):
    # make and initialize model
    config, ArgoDataset, collate_fn, net, loss, post_process, opt = lanegcn.get_model()
    ckpt_path = "36.000.ckpt"
    root_path = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(root_path, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    net.eval()
    map_net = net.map_net

    # get the features
    sample_node_features, sample_node_ctrs = [], []
    for i, list_of_ways in tqdm(enumerate(sample_ways)):
        graph = get_ubers_graph(list_of_ways)

        graph_batch = [graph]
        with torch.no_grad():
            nodes, node_idcs, node_ctrs = map_net(lanegcn.graph_gather(to_long(gpu(from_numpy(graph_batch)))))
        nodes = nodes.cpu().numpy()
        sample_node_features.append(nodes)
        sample_node_ctrs.append(node_ctrs)
    return sample_node_ctrs, sample_node_features


class KTreeNode:
    def __init__(self):
        self.child_nodes = []
        self.kmeans = None
        self.isLeaf = False
        self.sample_ids = set()  # only used for leaf nodes

cnt = 0
def make_k_tree(root, k, feature_points, lvl=0):
    global cnt
    cnt += 1
    print("doing node number", cnt, end=" ")
    if len(feature_points) < k or lvl >= 2:
        root.isLeaf = True
        return
    tic = time.time()
    root.kmeans = KMeans(k)
    cluster_ids = root.kmeans.fit_predict(feature_points)
    print("took", int(tic - time.time()), "seconds")
    for i in range(k):
        child_node = KTreeNode()
        root.child_nodes.append(child_node)
        make_k_tree(child_node, k, feature_points[cluster_ids == i], lvl+1)


def fill_k_tree(root, k, feature, sample_id, get_node=False):
    cur_node = root
    while not cur_node.isLeaf:
        cur_node = cur_node.child_nodes[cur_node.kmeans.predict([feature])[0]]
    if not get_node:
        cur_node.sample_ids.add(sample_id)
    else:
        return cur_node


def get_sample_ids_from_k_tree(root, feature):
    cur_node = root
    while not cur_node.isLeaf:
        cur_node = cur_node.child_nodes[cur_node.kmeans.predict([feature])[0]]
    return cur_node.sample_ids


def get_feature_cluster_id(root, feature, k):
    child_id = root.kmeans.predict([feature])[0]
    if root.child_nodes[child_id].isLeaf:
        return child_id
    return child_id * k + get_feature_cluster_id(root.child_nodes[child_id], feature, k)


def find_query_scene_similars_id(query_features, num_samples, root, n_retrieve):
    similar_feature_counts = np.zeros(num_samples)
    for feature in query_features:
        sample_ids = get_sample_ids_from_k_tree(root, feature)
        similar_feature_counts[list(sample_ids)] += 1
    return similar_feature_counts.argsort()[-n_retrieve:]


if __name__ == '__main__':
    min_xs, min_ys , max_xs, max_ys = {}, {}, {}, {}
    min_xs["NewMexico"], min_ys["NewMexico"], _, _ = utm.from_latlon(19.272, -99.344)
    max_xs["NewMexico"], max_ys["NewMexico"], _, _ = utm.from_latlon(19.584,  -98.849)
    min_xs["NewYork"], min_ys["NewYork"], _, _ = utm.from_latlon(40.634, -74.031)
    max_xs["NewYork"], max_ys["NewYork"], _, _ = utm.from_latlon(40.789, -73.726)
    min_xs["Paris"], min_ys["Paris"], _, _ = utm.from_latlon(48.788, 2.18)
    max_xs["Paris"], max_ys["Paris"], _, _ = utm.from_latlon(48.934, 2.513)
    min_xs["HongKong"], min_ys["HongKong"], _, _ = utm.from_latlon(22.205, 114.118)
    max_xs["HongKong"], max_ys["HongKong"], _, _ = utm.from_latlon(22.297, 114.267)

    x_size, y_size = 200, 200
    num_samples = 20000
    num_queries = 2
    batch_size = 1
    k = 10
    n_retrieve = 100
    save_dir = os.path.join("retrieved_scenario_images", "Run_Time_" + str(int(time.time())))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    load_from_saved_tree = False
    tic = time.time()
    with open(get_NewMexico_way_node_locations(), "rb") as f:
        way_node_locations_NewMexico = pickle.load(f)
    with open(get_HongKong_way_node_locations(), "rb") as f:
        way_node_locations_HongKong = pickle.load(f)
    with open(get_Paris_way_node_locations(), "rb") as f:
        way_node_locations_Paris = pickle.load(f)
    with open(get_NewYork_way_node_locations(), "rb") as f:
        way_node_locations_NewYork = pickle.load(f)

    way_node_locations = {
        "NewMexico": way_node_locations_NewMexico,
        "HongKong": way_node_locations_HongKong,
        "Paris": way_node_locations_Paris,
        "NewYork": way_node_locations_NewYork
    }
    if not args.load_saved_tree:
        tic = time.time()
        sample_ways, rect_ur_coords = get_samples(way_node_locations["NewYork"], min_xs["NewYork"], min_ys["NewYork"], max_xs["NewYork"], max_ys["NewYork"], x_size, y_size, num_samples)
        for city in way_node_locations.keys():
            if city != "NewYork":
                additional_sample_ways, additional_rects = get_samples(way_node_locations[city], min_xs[city], min_ys[city], max_xs[city], max_ys[city], x_size, y_size, num_samples)
                sample_ways = sample_ways + additional_sample_ways
                rect_ur_coords = np.concatenate([rect_ur_coords, additional_rects], 0)
        print("getting samples took {} seconds".format(time.time() - tic))

        tic = time.time()
        node_ctrs, node_features = get_ubers_node_features(sample_ways)
        all_features = np.concatenate(node_features, 0)
        print("extracting features took {} seconds".format(time.time() - tic))

        tic = time.time()
        ktree_root = KTreeNode()
        make_k_tree(ktree_root, k, all_features)
        print("constructing tree took {} seconds".format(time.time() - tic))

        tic = time.time()
        inputs = []
        for i, sample_features in tqdm(enumerate(node_features)):
            for feature in sample_features:
                fill_k_tree(ktree_root, k, feature, i)
        print("filling tree took {} seconds".format(time.time() - tic))
        with open(os.path.join(save_dir, "tree_root.pkl"), "wb") as f:
            pickle.dump(ktree_root, f)
        with open(os.path.join(save_dir, "sample_ways.pkl"), "wb") as f:
            pickle.dump(sample_ways, f)
        with open(os.path.join(save_dir, "rect_ur_coords.pkl"), "wb") as f:
            pickle.dump(rect_ur_coords, f)
        with open(os.path.join(save_dir, "node_ctrs.pkl"), "wb") as f:
            pickle.dump(node_ctrs, f)
        with open(os.path.join(save_dir, "node_features.pkl"), "wb") as f:
            pickle.dump(node_features, f)
    else:
        load_dir = get_K_Tree_load_dir()
        with open(os.path.join(load_dir, "tree_root.pkl"), "rb") as f:
            ktree_root = pickle.load(f)
        with open(os.path.join(load_dir, "sample_ways.pkl"), "rb") as f:
            sample_ways = pickle.load(f)
        with open(os.path.join(load_dir, "rect_ur_coords.pkl"), "rb") as f:
            rect_ur_coords = pickle.load(f)
        with open(os.path.join(load_dir, "node_ctrs.pkl"), "rb") as f:
            node_ctrs = pickle.load(f)

    num_samples *= len(way_node_locations.keys())
    tic = time.time()
    query_ways, query_rect_ur_coords = get_samples(way_node_locations["NewYork"], min_xs["NewYork"], min_ys["NewYork"], max_xs["NewYork"], max_ys["NewYork"], x_size, y_size, num_queries)
    # query_ways = []
    for i in range(10, 14):
        query_rect_ur_coords = np.concatenate([query_rect_ur_coords, np.array([[x_size, y_size]])])
        with open("Data/" + str(i) + ".pkl", 'rb') as f:
            query_ways.append(pickle.load(f))
    query_node_ctrs, query_node_features = get_ubers_node_features(query_ways)
    for i, query_features in enumerate(query_node_features):
        retrieved_ids = find_query_scene_similars_id(query_features, num_samples, ktree_root, n_retrieve)
        retrieved_scene_ctrs = []
        for id in retrieved_ids:
            retrieved_scene_ctrs.append(node_ctrs[id])

        save_addr = os.path.join(save_dir, str(i) + ".pdf")

        fig, ax = plt.subplots(4, 4)
        fig.set_size_inches(15, 15)
        plt.tight_layout()
        ax[0, 0].set_title("Query Scene")
        visualize_roads(query_ways[i], query_rect_ur_coords[i, 0] - x_size, query_rect_ur_coords[i, 1] - y_size,
                        query_rect_ur_coords[i, 0], query_rect_ur_coords[i, 1], ax[0, 0])

        print("searching for query number", i, "resulted in", retrieved_ids[:15], "retrievals")
        for ii, ind in enumerate([(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]):
            ax[ind].set_title("Retrieved Scene number " + str(ii+1) + ":" + str(retrieved_ids[ii]))
            retrieved_id = retrieved_ids[ii]
            visualize_roads(sample_ways[retrieved_id], rect_ur_coords[retrieved_id, 0] - x_size, rect_ur_coords[retrieved_id, 1] - y_size,
                        rect_ur_coords[retrieved_id, 0], rect_ur_coords[retrieved_id, 1], ax[ind])

        plt.savefig(save_addr, bbox_inches='tight')

    print("querying took {} seconds".format(time.time() - tic))


