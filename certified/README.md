# Certified Human Trajectory Prediction

Here, you can find the main codes of the paper and instructions on how to run them.

---

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Weights](#model-weights)
- [Evaluation](#evaluation)
  - [Original Baselines Evaluation](#1-original-baselines-evaluation)
  - [Certified Results Evaluation](#2-certified-results-evaluation)
  - [Visualization](#visualization)
- [Citation](#citation)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vita-epfl/s-attack.git
cd s-attack/certified
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 1.10
- NumPy, SciPy
- See `requirements.txt` for complete dependencies

---

## Data Preparation

This work uses the [TrajNet++](https://github.com/vita-epfl/trajnetplusplusdata/releases/tag/v4.0) dataset, which is shared between the `social-attack` and `certified` projects. The repository includes symbolic links from `certified/` to `social-attack/` for `DATA_BLOCK/` and `trajnetplusplustools/` to avoid data duplication.

**Expected structure after cloning:**
```
s-attack/
├── social-attack/
│   ├── DATA_BLOCK/
│   │   └── trajdata/
│   └── trajnetplusplustools/
└── certified/
    ├── DATA_BLOCK/              # symlink to ../social-attack/DATA_BLOCK
    ├── trajnetplusplustools/    # symlink to ../social-attack/trajnetplusplustools
    ├── baselines/
    └── ...
```

---

## Model Weights

Baseline models trained on TrajNet++ are required for evaluation. Download the weights from [here](https://drive.google.com/file/d/1ahizaqTJ4gFf6CwBGmJVIGw5US9g82BY/view?usp=sharing) and extract them to `baselines/weights/` in the certified directory.

**Note:** For the rights to use these baseline models, please consult the original authors.

**Weights structure:**
```
certified/
└── baselines/
    ├── weights/
    │   ├── Autobot/
    │   │   ├── Autobot_train.pkl.state
    │   ├── DPool/
    │   │   └── d_pool.state
    │   └── EqMotion/
    │       ├── my_checkpoint_v1.pth.tar
```

---

## Evaluation

### 1. Original Baselines Evaluation

Evaluate original (uncertified) baseline models on the TrajNet++:

```bash
python get_original_results.py
```

This computes `ADE`, `FDE`, and `collision` metrics for all baseline models. Results are saved in:
```
results/
├── d_pool/uncertified results.txt
├── autobot/uncertified results.txt
└── eq_motion/uncertified results.txt
```

**Note:** By default, evaluation runs on 1000 randomly sampled examples for faster testing. To evaluate on the full dataset, use `--max_samples -1`.

---

### 2. Certified Results Evaluation

Generate certified predictions with randomized smoothing:

```bash
# Run all three baselines (default)
python get_certfied_results.py
```


**Output metrics:**
- `ADE` / `FDE`: Average/Final Displacement Error
- `collision`: Collision rate
- `cert_collision`: Certified collision rate
- `FBD` / `ABD`: Final/Average Bound half-Diameter

Results are saved in:
```
results/
└── {model}/
    ├── mean - {denoiser} denoiser/
    │   └── sigma {σ} - r {r}.txt
    └── median - {denoiser} denoiser/
        └── sigma {σ} - r {r}.txt
```

**Performance Note:** This evaluation is computationally intensive as it tests multiple sigma values across all models. By default, the code evaluates on a smaller random subset of data to reduce runtime. To evaluate only a specific model, use `--model_types` (e.g., `--model_types eq_motion`).

---

### 3. Visualization

Generate comparison plots across baseline models:

```bash
python get_comparison_plots.py \
    --models d_pool autobot eq_motion \
    --smoothing_types mean median \
    --x_metric fde \
    --y_metric fbd \
    --target_r 0.1
```

**Arguments:**
- `--models`: Models to compare [default: all models]
- `--smoothing_types`: Smoothing operators [default: `mean`, `median`]
- `--x_metric` / `--y_metric`: Metrics to plot (see evaluation metrics above)
- `--target_r`: Certification radius [default: `0.1`]

Plots are saved in:
```
results/plots/
```

---

## Citation

If you find this work useful, please cite our paper:

```
@InProceedings{bahari2025certified,
    author    = {Bahari, Mohammadhossein and Saadatnejad, Saeed and Askari Farsangi, Amirhossein and Moosavi-Dezfooli, Seyed-Mohsen and Alahi, Alexandre},
    title     = {Certified Human Trajectory Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025},
}
```

---
