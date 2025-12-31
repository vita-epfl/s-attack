import torch
import argparse
import os
import numpy as np
from tqdm import tqdm

from dataset import Dataset
from model import Model
from utils import get_arguments, create_folders
from smooth_model import Evaluator


def evaluate_baseline(model_type, dataset, evaluator_args, device, max_samples=None):
    """Evaluate a single baseline model on the dataset."""
    
    print(f"\nEvaluating {model_type.upper()}...")
    if max_samples:
        print(f"Max Samples\t\t:\t{max_samples}")
    
    arguments = get_arguments()
    model = Model(model_type, arguments, device)
    
    metrics_str_all = "ade\tfde\tcollision\n"
    
    # Randomly subsample dataset if max_samples is specified
    if max_samples and max_samples > 0 and max_samples < len(dataset):
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset_subset = [dataset[i] for i in sorted(indices)]
    else:
        dataset_subset = dataset
    
    for data in tqdm(dataset_subset, desc=f"Processing {model_type}"):
        scene = data[1]
        pred = model.get_prediction_single_data(data)
        
        evaluator = Evaluator(evaluator_args)
        fde, ade = evaluator.calc_fde_ade(pred, scene[-evaluator.pred_length:])
        collision = evaluator.check_collision(pred, scene[-evaluator.pred_length:])
        
        metrics_str_all += f"{ade}\t{fde}\t{collision}\n"
    
    # Save results
    save_folder_path = create_folders("results", (f"{model_type}",))
    save_path = os.path.join(save_folder_path, "uncertified results.txt")
    
    with open(save_path, 'w') as f:
        f.write(metrics_str_all)
    
    print(f"✓ Results saved to: {save_path}")


if __name__ == "__main__":

    """
    Evaluate original (uncertified) baseline models on TrajNet++.
    
    Computes ADE, FDE, and collision metrics for specified baseline models.
    Results are saved in results/{model}/ directories.
    """

    parser = argparse.ArgumentParser(
        description='Evaluate original (uncertified) baseline models on TrajNet++'
    )
    
    parser.add_argument('--models', type=str, nargs='+',
                        default=['eq_motion', 'autobot', 'd_pool'],
                        choices=['eq_motion', 'autobot', 'd_pool'],
                        help='List of baseline models to evaluate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., cuda:0, cpu). If not specified, auto-detects GPU')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to evaluate (randomly subsampled). Default: 1000. Set to -1 to use all samples')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("Original Baselines Evaluation")
    print(f"Models\t\t:\t{args.models}")
    print(f"Device\t\t:\t{device}")
    
    # Load dataset
    arguments = get_arguments()
    dataset_args = arguments["dataset args"]
    
    print("\nLoading dataset...")
    dataset = Dataset(dataset_args)
    print(f"✓ Loaded {len(dataset)} scenes")
    
    evaluator_args = arguments["evaluator args"]
    
    # Evaluate each model
    for model_type in args.models:
        evaluate_baseline(model_type, dataset, evaluator_args, device, args.max_samples)
    
    print("✓ All evaluations completed!")
