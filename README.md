# Pulse2BeatGAN

This repository contains the implementation of **Pulse2BeatGAN**, a deep learning framework for estimating ECG signals from PPG signals. It supports multiple model architectures and datasets, enabling robust ablation studies and cross-dataset evaluations.

## Features

*   **Models**:
    *   `swin_unet_gab`: Swin Transformer U-Net with Guided Attention Blocks (Proposed).
    *   `swin_unet`: Standard Swin Transformer U-Net.
    *   `unet`: Standard U-Net Generator.
    *   `multires_unet`: Multi-Residual U-Net.
*   **Datasets**:
    *   BIDMC
    *   DALIA
    *   WESAD
    *   Capnobase
    *   MIMIC-AFib
    *   UQVitalSigns
*   **Losses**:
    *   Pixel-wise L1 Loss
    *   GAN Loss (WGAN-GP)
    *   Spectral Losses: 
        *   Bandwidth (IPR)
        *   **Dynamic Sparsity (SNR)**: Adapts to the signal's dominant frequency.
        *   Variance (EMD)
    *   Guided Attention Loss

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install torch torchvision torchaudio numpy pandas scipy scikit-learn matplotlib tqdm neurokit2 similaritymeasures
    ```

## Usage

### Training

To train a model, use `main.py` with the desired arguments:

```bash
python main.py --model [MODEL_NAME] --dataset [DATASET_NAME] --limit [LIMIT] --n_epochs [EPOCHS] --batch_size [BATCH_SIZE]
```

**Arguments:**
*   `--model`: Choose from `swin_unet_gab`, `swin_unet`, `unet`, `multires_unet`.
*   `--dataset`: Choose from `bidmc`, `dalia`, `wesad`, `capnobase`, `mimic`, `uqvitalsigns`.
*   `--limit`: (Optional) Limit the number of samples for quick testing.
*   `--n_epochs`: Number of training epochs.
*   `--batch_size`: Batch size.

**Example:**
```bash
python main.py --model swin_unet_gab --dataset bidmc --n_epochs 50 --batch_size 32
```

### Cross-Dataset Evaluation

To train on one dataset and validate on another:

```bash
python main.py --dataset bidmc --test_dataset wesad ...
```

## Project Structure

*   `src/models/`: Model implementations (`swin_unet_gab.py`, `swin_unet.py`, `variants.py`).
*   `src/data/`: Data loading and preprocessing (`dataset.py`, `preprocessing.py`).
*   `src/training/`: Training loop (`train.py`).
*   `src/utils/`: Utility functions, losses, and metrics (`losses.py`, `metrics.py`).
*   `main.py`: Entry point.

