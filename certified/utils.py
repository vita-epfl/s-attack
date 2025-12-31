import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle


def get_arguments():
    
    arguments = {}
    
    general_args = {'pred_length' : 12,
                    'obs_length' : 9,
                    }
    
    
    dataset_args = {"data_part" : "train",
                    'path' : "DATA_BLOCK/trajdata",
                    'sample' : 1.0,
                    "goals" : False,
                    }
    
    d_pool_args = {"pred_length" : general_args["pred_length"],
                   "obs_length" : general_args["obs_length"],
                   'coordinate_embedding_dim' : 64,
                   'hidden_dim' : 128,
                   'goal_dim' : 64,
                   "goals" : False,                   
                   'weights_path' : "baselines/weights/DPool/d_pool.state",
                   'pool_dim' : 256,
                   'vel_dim' : 32,
                   'neigh' : 4,
                   }
    
    autobot_args = {'weights_path' : "baselines/weights/Autobot/Autobot_train.pkl.state",
                    'pred_length' : general_args["pred_length"],
                    'obs_length' : general_args["obs_length"],
                    'modes' : 1,
                    'jobs number' : 1}
    
    eq_motion_args = {'weights_path' : "baselines/weights/EqMotion/my_checkpoint_v1.pth.tar"}
    
    smooth_model_args = {"clamping_margin" : 0,
                         "p" : 0.5,    
                         'n_monte_carlo' : 100,
                         'obs_length' : general_args["obs_length"],
                         'max_r_for_clamping' : 1
                         }
    
    evaluator_args = {"pred_length" : general_args["pred_length"],
                      'collision_threshold' : 0.2}
    
    polynomial_denoiser_args = {"polynomial_order" : 3}
    
    
    arguments["general args"] = general_args
    arguments["dataset args"] = dataset_args
    arguments["D-Pool args"] = d_pool_args
    arguments["Autobot args"] = autobot_args
    arguments["EqMotion args"] = eq_motion_args
    arguments["smooth model args"] = smooth_model_args
    arguments["evaluator args"] = evaluator_args
    arguments["polynomial denoiser args"] = polynomial_denoiser_args
    
    
    return arguments


class CurvePlotter:
    
    def __init__(self, x_label, y_label, title):
        
        self.fig, self.ax = plt.subplots(figsize = (12, 10))
        
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        
        self.marker_size = 7
        self.marker_size_uncert = 7
        self.marker_size_uncert_point = 15
        self.legend_marker_size_ratio = 1.5
        
        self.line_width = 3
        self.line_width_uncert = 2
        
        self.labels_font_size = 18
        self.labelpad = 15
        
        self.font_name = "DejaVu Sans" #"Helvetica Neue"
        self.legends_font_size = 18
        self.annotations_font_size = 15
        
        self.title_font_size = 22
                
        self.colors_dict, self.colors_list = self.get_colors()        
        self.markers_dict, self.markers_list = self.get_markers()
        self.color_idx = 0
        self.marker_idx = 0
        
        self.legend_handles = []
        
        self.uncertified_ranges = {"x" : None,
                                   "y" : None}
        
     
    def get_colors(self):
        
        colors_d_pool = {"wiener_filter" : "#FF6969",
                         None : "#69FFFB",
                         "moving_average" : "#77FF68",
                         "uncertified" : "#FF6969"}#"#FFF806"}
        
        colors_autobot = {"wiener_filter" : "#59BD60",
                          None : "#CD4D4D",
                          "moving_average" : "#CDC04D",
                          "uncertified" : "#59BD60"}#"#00F7FF"}
        
        colors_eq_motion = {"wiener_filter" : "#587FBC",
                            None : "#BD5959",
                            "moving_average" : "#BA59BD",
                            "uncertified" : "#587FBC"}#"#00FF6F",

        
        colors = {"eq_motion" : colors_eq_motion,
                  "d_pool" : colors_d_pool,
                  "autobot" : colors_autobot}
        
        colors_set = []
        for k in colors.keys():
            for v in colors[k].values():
                if v not in colors_set:
                    colors_set.append(v)
        
        return colors, colors_set
    
    
    def get_markers(self):
        
        markers_d_pool = {"wiener_filter" : "o",
                         None : "X",
                         "moving_average" : "2",
                         "uncertified" : "h"}
        
        markers_autobot = {"wiener_filter" : "^",
                          None : "X",
                          "moving_average" : "2",
                          "uncertified" : "D"}
        
        markers_eq_motion = {"wiener_filter" : "s",
                            None : "X",
                            "moving_average" : "2",
                            "uncertified" : "*"}

        
        
        markers = {"eq_motion" : markers_eq_motion,
                  "d_pool" : markers_d_pool,
                  "autobot" : markers_autobot}
        
        
        markers_list = ["o", "X", "^", "D", "s", "*", "h", "<", "|", "P"]
        
        
        
        return markers, markers_list
    
    
        
        
    
    
    def plot_one_curve(self, x_data, y_data, sigmas, model_type, denoiser_type, label, use_property_dict = True, add_annotations = True):
        
        if use_property_dict:
            color = self.colors_dict[model_type][denoiser_type]
            marker = self.markers_dict[model_type][denoiser_type]
        else:
            color = self.colors_list[self.color_idx]
            self.color_idx = (self.color_idx + 1) % len(self.colors_list)
            marker = self.markers_list[self.marker_idx]
            self.marker_idx = (self.marker_idx + 1) % len(self.markers_list)
        
        
        self.ax.plot(x_data,
                     y_data,
                     marker = marker,
                     markersize = self.marker_size,
                     linewidth = self.line_width,
                     color = color)
        
        legend_marker_size = self.marker_size * self.legend_marker_size_ratio
        legend_handle = Line2D([0],
                               [0],
                               marker = marker,
                               color='w',
                               label = label,
                               markerfacecolor = color,
                               markersize = legend_marker_size)
        if add_annotations:
            self.insert_annotations(x_data, y_data, sigmas)
        self.legend_handles.append(legend_handle)
    
    def insert_annotations(self, x_data, y_data, sigmas):
        
        if self.x_label != "sigma" and self.y_label != "sigma":
        
            for i in [0, -1]:
                
                x = x_data[i]
                y = y_data[i]
                
                sigma_value = round(sigmas[i], 2)
                
                annotation = r"$\sigma = {}$".format(sigma_value)
                
                self.ax.annotate(annotation,
                                 (x,y),
                                 textcoords="offset points",
                                 xytext=(50,10),
                                 ha='right',
                                 fontsize = self.annotations_font_size,
                                 fontname =  self.font_name,
                                 weight = "light")
        
    def plot_uncertified_results(self, x_data, y_data, model_type, label, use_property_dict = True):
        
        if use_property_dict:
            color = self.colors_dict[model_type]["uncertified"]
            marker = self.markers_dict[model_type]["uncertified"]
        else:
            color = self.colors_list[self.color_idx]
            self.color_idx = (self.color_idx + 1) % len(self.colors_list)
            marker = self.markers_list[self.marker_idx]
            self.marker_idx = (self.marker_idx + 1) % len(self.markers_list)
        
        if x_data is None and y_data is None:
            return
        
        elif x_data is None and y_data is not None:            
            marker_size = self.marker_size_uncert
            min_x, max_x = self.get_uncertified_range("x")
            self.ax.hlines(y_data, min_x, max_x,
                           linestyle = "dashed",
                           linewidth = self.line_width_uncert,
                           color = color)
            
            legend_handle = Line2D([0, 0.5],
                                   [0, 0],
                                   linestyle = "dashed",
                                   linewidth = self.line_width_uncert,
                                   color = color,
                                   label = label)
            
        elif x_data is not None and y_data is None:
            
            
            marker_size = self.marker_size_uncert
            min_y, max_y = self.get_uncertified_range("y")
            self.ax.vlines(x_data, min_y, max_y,
                           linestyle = "dashed",
                           linewidth = self.line_width_uncert,
                           color = color)
            
            legend_handle = Line2D([0, 0],
                                   [0, 0.5],
                                   linestyle = "dashed",
                                   linewidth = self.line_width_uncert,
                                   color = color,
                                   label = label)
            
        elif x_data is not None and y_data is not None:            
            
            marker_size = self.marker_size_uncert_point
            legend_marker_size = marker_size * self.legend_marker_size_ratio
            self.ax.plot(x_data,
                         y_data,
                         marker = marker,
                         markersize = marker_size,
                         color = color)
        
            legend_handle = Line2D([0],
                                   [0],
                                   marker = marker,
                                   color='w',
                                   label = label,
                                   markerfacecolor = color,
                                   markersize = legend_marker_size)
        
        self.legend_handles.append(legend_handle)
        
    def get_uncertified_range(self, axis):
        
        if self.uncertified_ranges[axis] is not None:
            return self.uncertified_ranges[axis]
        
        margin = 0.05
        
        if axis == "x":
            init_range = self.ax.get_xlim()
        elif axis == "y":
            init_range = self.ax.get_ylim()
        
        low = init_range[0]
        high = init_range[1]
        
        diff = high - low
        low = low - margin * diff
        high = high + margin * diff
        
        self.uncertified_ranges[axis] = (low, high)
        
        return low, high
        
    
    def adjust_plot(self, legend_columns_number = 3):
        
        max_col = legend_columns_number
        
        n = len(self.legend_handles)
        ncol = min(max_col, n)
        nrow = np.ceil(n / max_col)
                
        legend_bbox_to_anchor = (0, 1 + 0.1 * nrow * self.legends_font_size / 22)
        
        legend_font_props = FontProperties(family = self.font_name,
                                           size = self.legends_font_size)
        
        
        plt.legend(loc='upper left',
                   bbox_to_anchor = legend_bbox_to_anchor,
                   ncol = ncol,
                   prop = legend_font_props,
                   frameon=False,
                   handles = self.legend_handles)
        
        
        self.ax.set_xlabel(self.x_label,
                           fontsize = self.labels_font_size,
                           fontname = self.font_name,
                           labelpad = self.labelpad)
        
        self.ax.set_ylabel(self.y_label,
                           fontsize = self.labels_font_size,
                           fontname = self.font_name,
                           labelpad = self.labelpad)
        
        title_pad = 10 + 40 * nrow
        
        self.ax.set_title(self.title,
                          fontsize = self.title_font_size,
                          fontname = self.font_name,
                          pad = title_pad) 
        
        plt.grid('on')
        plt.tight_layout()


class TrajectoryPlotter:
    
    def __init__(self, ax, title):
        
        self.ax = ax
        self.title = title
        
        self.box_face_color = "#AAFFFF"
        self.box_edge_color = "#55AAAA"
        self.box_alpha = 0.2
        
        self.font_name = "DejaVu Sans" #"Helvetica Neue"
        self.title_font_size = 16
    
    
    def plot_past(self, xy , color, label , alpha = 1):
        xy_plot = xy.clone().detach().cpu()
        self.ax.plot(xy_plot[:, 0], xy_plot[:, 1],
                     color = color,
                     marker = '.',
                     alpha = alpha,
                     label = label)
    
    def plot_future(self, xy, past_xy, color, label, alpha = 1):
        
        xy_plot = xy.clone().detach().cpu()
        past_xy_plot = past_xy.clone().detach().cpu()
        
        self.ax.plot(xy_plot[:, 0], xy_plot[:, 1],
                     color = color,
                     marker = '.',
                     alpha = alpha,
                     linestyle = 'dashed',
                     label = label)
        
        self.ax.plot([past_xy_plot[-1, 0] , xy_plot[0, 0]],
                     [past_xy_plot[-1, 1] , xy_plot[0, 1]],
                     color = color,
                     marker = '.',
                     alpha = alpha,
                     linestyle = 'dashed')
        
        
    def plot_boxes(self, certified_bounds, label):
        
        lower_b, upper_b = certified_bounds[:, :, 0, :]
        lower_b = lower_b.detach().cpu()
        upper_b = upper_b.detach().cpu()
        for k in range(upper_b.shape[0]):
            x = lower_b[k, 0]
            y = lower_b[k, 1]
            width = upper_b[k, 0] - lower_b[k, 0]
            height = upper_b[k, 1] - lower_b[k, 1]
                    
            self.ax.add_patch(Rectangle((x, y), width, height, 
                                   facecolor = self.box_face_color,
                                   alpha = self.box_alpha,
                                   edgecolor = self.box_edge_color,
                                   label = label if k == 0 else None))



    def adjust(self):
        
        self.ax.set_title(self.title,
                          fontsize = self.title_font_size,
                          fontname = self.font_name)
        
        plt.setp(self.ax.get_xticklabels(), fontsize = self.title_font_size)
        plt.setp(self.ax.get_yticklabels(), fontsize = self.title_font_size)
        
        self.ax.set_aspect('equal')
        



def create_folders(parent_folder, sub_folders):
    
    path = parent_folder
    if not os.path.exists(path):
        os.mkdir(path)
    
    for sub_folder in sub_folders:
        path = os.path.join(path, sub_folder)
        if not os.path.exists(path):
            os.mkdir(path)
    
    return path


def find_unexplored_r_sigma(all_rs, all_sigmas, folder_path):
    
    threshold = 0.0001
    
    all_pairs = []
    for r in all_rs:
        for sigma in all_sigmas:
            all_pairs.append((r, sigma))
    
    explored_pairs = []
    file_names = os.listdir(folder_path)
    for name in file_names:
        sigma = float(name[:-4].split(" - ")[0].split(" ")[1])
        r = float(name[:-4].split(" - ")[1].split(" ")[1])
        
        explored_pairs.append((r, sigma))
    
    unexplored_pairs = []
    
    for r, sigma in all_pairs:
        is_explored = False
        for er, es in explored_pairs:
            if abs(r - er) < threshold and abs(sigma - es) < threshold:
                is_explored = True
        
        if not is_explored:
            unexplored_pairs.append((r, sigma))
        
    return unexplored_pairs


def load_one_metric_data(metric, list_files_path):
    
    data = []
    
    for path in list_files_path:
        df = pd.read_csv(path, sep="\t")
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        data.append(df[metric].mean())
    
    return data


def load_one_model_data(parent_folder,
                        x_metric,
                        y_metric):
    
    r_files = {}
    for name in os.listdir(parent_folder):
        r = float(name[:-4].split(" - ")[1].split(" ")[1])
        if r not in r_files.keys():
            r_files[r] = []
        
        r_files[r].append(name)
    
    
    
    data_dict = {}
    for r in r_files.keys():
        
        files_path = [os.path.join(parent_folder, name) for name in r_files[r]]
        
        x_data = load_one_metric_data(x_metric, files_path)
        y_data = load_one_metric_data(y_metric, files_path)
        sigmas = load_one_metric_data("sigma", files_path)
        
        sigmas, x_data, y_data = zip(*(sorted(zip(sigmas, x_data, y_data))))
        
        data_dict[r] = (sigmas, x_data, y_data)
        
    
    return data_dict
    
    


def load_uncertified_data(x_metric,
                          y_metric,
                          model_type):
    
    acceptable_uncertified = ["ade", "fde", "collision"]
    
    path = os.path.join("results/",
                        f"{model_type}",
                        "uncertified results.txt")
    x_data_uncert = None
    y_data_uncert = None
    if x_metric in acceptable_uncertified:
        x_data_uncert = load_one_metric_data(x_metric, [path])
    
    if y_metric in acceptable_uncertified:
        y_data_uncert = load_one_metric_data(y_metric, [path])
    
    
    return x_data_uncert, y_data_uncert