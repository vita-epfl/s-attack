import os
argo_path = "/work/vita/datasets/argoverse/argoverse-api/"
lanegcn_model_weights_path = "./36.000.ckpt"
datf_model_weights_path = "/home/mshahver/CMU-code/code/experiment/attgscam.argo"
wimp_model_weights_path = "/home/mshahver/rl-scene-attack/attackedmodels/WIMP_master/experiments/example2/checkpoints/"
argo_preprocessed_data_path = "/scratch/izar/mshahver/val_crs_dist6_angle90.p"
argo_val_preprocessed_data_path = "/scratch/izar/mshahver/val_crs_dist6_angle90.p"
syn_data_train_path = "/scratch/izar/mshahver/train_data_full.pkl"
syn_data_val_path = "/scratch/izar/mshahver/val_data_full.pkl"
NewMexico_way_node_locations = "Data/NewMexico_way_node_locations.pkl"
HongKong_way_node_locations = "Data/HongKong_way_node_locations.pkl"
Paris_way_node_locations = "Data/Paris_way_node_locations.pkl"
NewYork_way_node_locations = "Data/NewYork_way_node_locations.pkl"
K_Tree_load_dir = ""


def get_K_Tree_load_dir():
    return K_Tree_load_dir


def get_NewMexico_way_node_locations():
    return NewMexico_way_node_locations


def get_HongKong_way_node_locations():
    return HongKong_way_node_locations


def get_Paris_way_node_locations():
    return Paris_way_node_locations


def get_NewYork_way_node_locations():
    return NewYork_way_node_locations


def get_syn_data_train():
    return syn_data_train_path


def get_syn_data_val():
    return syn_data_val_path


def get_argo_val_path():
    return os.path.join(argo_path, "val/data")


def get_argo_path():
    return argo_path


def get_argo_preprocessed():
    return argo_preprocessed_data_path


def get_argo_val_preprocessed():
    return argo_val_preprocessed_data_path


def get_lanegcn_path():
    return lanegcn_model_weights_path


def get_datf_path():
    return datf_model_weights_path


def get_wimp_path():
    return wimp_model_weights_path