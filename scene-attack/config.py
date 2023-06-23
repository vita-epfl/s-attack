import os
argo_path = ""
lanegcn_model_weights_path = "./36.000.ckpt"
datf_model_weights_path = ""
wimp_model_weights_path = ""
argo_preprocessed_data_path = ""
argo_val_preprocessed_data_path = ""
syn_data_train_path = ""
syn_data_val_path = ""
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
