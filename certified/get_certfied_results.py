import torch
import numpy as np
import argparse

from dataset import Dataset
from smooth_model import SmoothModel, Evaluator
from model import Model
from utils import get_arguments, find_unexplored_r_sigma, create_folders
import os

from tqdm import tqdm

def run_pipeline(model_type,
                 denoiser_type,
                 device,
                 all_rs,
                 all_sigmas,
                 max_samples=None):
    
    print("--------------------------------------")
    print("Model Type\t\t:\t{}".format(model_type))
    print("Device\t\t\t:\t{}".format(device))
    print("Denoiser\t\t:\t{}".format(denoiser_type))
    if max_samples:
        print("Max Samples\t\t:\t{}".format(max_samples))
    
    arguments = get_arguments()
    
    
    
    dataset_args = arguments["dataset args"]
    
    smooth_model_args = arguments["smooth model args"]    

    evaluator_args = arguments["evaluator args"]
    
    
    
    dataset = Dataset(dataset_args)
    
    model = Model(model_type,
                  arguments,
                  device)
    
    
    parent_folder = "results"
    all_path = {}
    
    for smoothing_type in ["mean", "median"]:        
        
        sub_folders = ("{}".format(model_type),
                       f"{smoothing_type} - {denoiser_type} denoiser")
        path = create_folders(parent_folder, sub_folders)
        all_path[smoothing_type] = path
    
    unexplored_r_sigma = find_unexplored_r_sigma(all_rs,
                                                 all_sigmas,
                                                 folder_path = all_path["mean"])
    
    print("\nTotal Unexplored Sigmas\t:\t{}\n".format(len(unexplored_r_sigma)))
    
    with torch.no_grad():
    
        for r, sigma in unexplored_r_sigma:
            
            print(r, sigma)    
            
            smooth_model = SmoothModel(model,
                                       denoiser_type,
                                       r,
                                       sigma,
                                       smooth_model_args)
                            
            evaluator = Evaluator(evaluator_args)
            
            metrics_str_all = {}
            
            for smoothing_type in ["mean", "median"]:
                metrics_str_all[smoothing_type] = "r\tsigma\t" + evaluator.header_str
            
            # Randomly subsample dataset if max_samples is specified
            if max_samples and max_samples > 0 and max_samples < len(dataset):
                indices = np.random.choice(len(dataset), max_samples, replace=False)
                dataset_subset = [dataset[i] for i in sorted(indices)]
            else:
                dataset_subset = dataset
            
            for data in tqdm(dataset_subset):
                
                smoothed_pred_all, certified_bounds_all = smooth_model.get_smoothed_results(data)
                
                
                for operator in ["mean", "median"]:
                    temp_str = f"{r}\t{sigma}\t"
                    temp_str += evaluator.compute_metrics(
                        data[1],
                        smoothed_pred_all[operator],
                        certified_bounds_all[operator]
                        )
                    metrics_str_all[operator] += temp_str

            
            for operator in ["mean", "median"]:        
                file_name = "sigma {} - r {}.txt".format(sigma, r)
                save_path = os.path.join(all_path[operator], file_name)
                with open(save_path, 'w') as f:
                    f.write(metrics_str_all[operator])
                
                    
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get certified results for trajectory prediction')
    parser.add_argument('--model_types', type=str, nargs='+', default=['eq_motion', 'autobot', 'd_pool'], 
                        choices=['eq_motion', 'autobot', 'd_pool'],
                        help='Type(s) of model to use (default: all three baselines)')
    parser.add_argument('--denoiser_type', type=str, default='wiener_filter',
                        choices=['wiener_filter', 'moving_average', 'polynomial'],
                        help='Type of denoiser to use')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., cuda:0, cpu). If not specified, auto-detects GPU')
    parser.add_argument('--all_rs', type=float, nargs='+', default=[0.1],
                        help='List of r values for certification (default: [0.1])')
    parser.add_argument('--all_sigmas', type=float, nargs='+', 
                        default=[round(x, 3) for x in np.linspace(0.08, 0.4, 9)],
                        help='List of sigma values (default: 9 values from 0.08 to 0.4)')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to evaluate (randomly subsampled). Default: 1000. Set to -1 to use all samples')
    
    args = parser.parse_args()
    
    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    for model_type in args.model_types:
        run_pipeline(model_type,
                     args.denoiser_type,                               
                     device,
                     args.all_rs,
                     args.all_sigmas,
                     args.max_samples)
    
