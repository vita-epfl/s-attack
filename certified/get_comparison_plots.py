import os
import argparse
from utils import create_folders, load_one_model_data, load_uncertified_data, CurvePlotter


def ploting_function(x_metric,
                     y_metric,
                     all_models,
                     denoiser_type,
                     smoothing_type,
                     target_r):

    
    save_folder_path = create_folders("results", ("plots",))
    
    title = f"{denoiser_type} denoiser - {smoothing_type} smoothing - r = {target_r}"
    
    plotter = CurvePlotter(x_metric, y_metric, title)
    
    
    for model_type in all_models:
        
        
        parent_folder = os.path.join("results/",
                                     f"{model_type}",
                                     f"{smoothing_type} - {denoiser_type} denoiser")
        
        data_dict = load_one_model_data(parent_folder,
                                        x_metric,
                                        y_metric)
        
        
        
        sigmas, x_data, y_data = data_dict[target_r]
        plotter.plot_one_curve(x_data,
                               y_data,
                               sigmas,
                               model_type,
                               denoiser_type,
                               f"{model_type}")
        
        x_data_uncert, y_data_uncert = load_uncertified_data(x_metric, y_metric, model_type)

        plotter.plot_uncertified_results(x_data_uncert,
                                         y_data_uncert,
                                         model_type,
                                         f"Original {model_type}")
    
    plotter.adjust_plot()
    
    name = f"baselines - {denoiser_type} denoiser - {smoothing_type} - r {target_r} - {y_metric} vs {x_metric}"
    save_path = os.path.join(save_folder_path, f"{name}.png")
    
    plotter.fig.savefig(save_path, dpi = 300)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate comparison plots for trajectory prediction models')
    
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['d_pool', 'autobot', 'eq_motion'],
                        choices=['d_pool', 'autobot', 'eq_motion'],
                        help='List of models to compare')
    parser.add_argument('--denoisers', type=str, nargs='+',
                        default=['wiener_filter'],
                        choices=['wiener_filter', 'moving_average', 'polynomial'],
                        help='List of denoisers to plot')
    parser.add_argument('--smoothing_types', type=str, nargs='+',
                        default=['median'],
                        choices=['mean', 'median'],
                        help='List of smoothing types to plot')
    parser.add_argument('--target_r', type=float, default=0.1,
                        help='Target r value for certification')
    parser.add_argument('--x_metric', type=str, default='fbd',
                        choices=["sigma", "r", "ade", "fde", "fbd", "abd",
                                    "collision", "cert_collision"],
                        help='Metric for x-axis')
    parser.add_argument('--y_metric', type=str, default='fde',
                        choices=["sigma", "r", "ade", "fde", "fbd", "abd",
                                    "collision", "cert_collision"],
                        help='Metric for y-axis')
    
    args = parser.parse_args()
    
    print(f"X metric\t:\t{args.x_metric}")
    print(f"Y metric\t:\t{args.y_metric}")
    print(f"Models\t\t:\t{args.models}")
    print(f"Denoisers\t:\t{args.denoisers}")
    print(f"Smoothing\t:\t{args.smoothing_types}")
    print(f"Target r\t:\t{args.target_r}")
    
    for denoiser in args.denoisers:
        for smoothing_type in args.smoothing_types:
            print(f"\nGenerating plot for {denoiser} denoiser, {smoothing_type} smoothing...")
            ploting_function(args.x_metric,
                           args.y_metric,
                           args.models,
                           denoiser,
                           smoothing_type,
                           args.target_r)