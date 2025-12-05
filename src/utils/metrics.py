import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity
import logging
import similaritymeasures
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Helper for smoother
mean_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, bias=False, padding_mode='replicate', padding=1)
mean_conv.weight.data = torch.full_like(mean_conv.weight.data, 0.25)
pad = nn.ReplicationPad1d((0, 1))

def smoother(tensor, device):
    tensor = mean_conv.to(device)(tensor)
    tensor = pad.to(device)(tensor)
    return tensor

def eval_metrics(signal_a, signal_b):
    """
    Evaluate metrics between generated and real signals.
    """
    rmse = np.sqrt(((signal_a - signal_b) ** 2).mean())
    
    std_a = np.std(signal_a)
    std_b = np.std(signal_b)
    
    if std_a == 0 or std_b == 0:
        logging.warning(f"Zero variance detected. std_a: {std_a}, std_b: {std_b}")
        p = np.nan
        r_squared = np.nan
        valid = False
    else:
        try:
            p, _ = stats.pearsonr(signal_a, signal_b)
            ss_res = ((signal_a - signal_b) ** 2).sum()
            ss_tot = ((signal_a - np.mean(signal_a)) ** 2).sum()
            if ss_tot == 0:
                logging.warning("Sum of squares total is zero, setting r_squared to nan.")
                r_squared = np.nan
                valid = False
            else:
                r_squared = 1 - ss_res / ss_tot
                valid = True
        except Exception as e:
            logging.error(f"Error computing metrics: {e}")
            p = np.nan
            r_squared = np.nan
            valid = False
    
    return rmse, p, r_squared, valid

def PSNR(pred, target):
    """
    Compute the Peak Signal-to-Noise Ratio between predicted and target signals.
    """
    # Ensure that pred and target are 2D arrays
    assert pred.ndim == 2 and target.ndim == 2, "Pred and Target must be 2D arrays"

    psnr_values = []
    for i in range(pred.shape[0]):  # Loop over the number of samples
        pred_single = pred[i, :]
        target_single = target[i, :]

        mse = np.mean((pred_single - target_single) ** 2)
        if mse == 0:
            psnr = float('inf')
            logging.debug(f"Sample {i}: MSE is zero, PSNR is infinite.")
        else:
            max_target = np.max(np.abs(target_single))  # Use absolute to account for negative peaks
            if max_target == 0:
                psnr = np.nan
                logging.warning(f"Sample {i}: max_target is zero, PSNR is undefined.")
            else:
                psnr = 20 * np.log10(max_target / (np.sqrt(mse) + 1e-10))
        psnr_values.append(psnr)

    psnr_values = np.array(psnr_values)
    valid_psnr = psnr_values[np.isfinite(psnr_values)]

    if valid_psnr.size == 0:
        psnr_mean = np.nan
        psnr_std = np.nan
        logging.warning("All PSNR values are invalid (nan or inf).")
    else:
        psnr_mean = np.mean(valid_psnr)
        psnr_std = np.std(valid_psnr)

    return psnr_mean, psnr_std

def SSIM(pred, gt):
    """
    Compute the Structural Similarity Index Measure between predicted and ground truth signals.
    """
    # Ensure pred and gt are 2D arrays
    assert pred.ndim == 2 and gt.ndim == 2, "Pred and GT must be 2D arrays"

    ssim_values = []
    for i in range(pred.shape[0]):  # Loop over the number of samples
        pred_single = pred[i, :]
        gt_single = gt[i, :]

        data_range = np.max(gt_single) - np.min(gt_single)
        if data_range == 0:
            ssim = np.nan
            logging.warning(f"Sample {i}: data_range is zero, SSIM is undefined.")
        else:
            ssim = structural_similarity(pred_single, gt_single, data_range=data_range)
        ssim_values.append(ssim)

    ssim_values = np.array(ssim_values)
    valid_ssim = ssim_values[np.isfinite(ssim_values)]

    if valid_ssim.size == 0:
        ssim_mean = np.nan
        ssim_std = np.nan
        logging.warning("All SSIM values are invalid (nan or inf).")
    else:
        ssim_mean = np.mean(valid_ssim)
        ssim_std = np.std(valid_ssim)

    return ssim_mean, ssim_std

from src.data.preprocessing import create_peak_mask

def evaluate_generated_signal_quality(val_dataloader, ecg_generator, writer, steps, device):
    """
    Evaluate the quality of generated ECG signals against real signals using multiple metrics.
    Logs and returns the evaluation results.
    """
    ecg_generator.eval()

    all_real_ecg = []
    all_generated_ecg = []
    
    # We iterate through the dataloader
    # Note: The batch structure in train.py is:
    # batch[0]: ppg_128
    # batch[1]: ppg_256
    # batch[2]: ppg
    # batch[3]: ecg_128
    # batch[4]: ecg_256
    # batch[5]: ecg
    
    for batch_idx, batch in enumerate(val_dataloader):
        real_ppg = batch[2].unsqueeze(dim=1).to(device).float()
        real_ecg = batch[5].unsqueeze(dim=1).to(device).float()
        
        with torch.no_grad():
            if ecg_generator.__class__.__name__ == 'SwinTransformerSysGAB':
                 # Generate masks
                 rpeaks_masks = []
                 for sig in real_ecg.cpu().numpy():
                     mask = create_peak_mask(sig.flatten(), sampling_rate=128)
                     rpeaks_masks.append(mask)
                 rpeaks_masks = torch.tensor(np.array(rpeaks_masks)).float().unsqueeze(1).to(device)
                 opeaks_masks = torch.zeros_like(rpeaks_masks).to(device)
                 
                 _, _, fake_ecg, _ = ecg_generator(real_ppg, rpeaks_masks, opeaks_masks)
            else:
                outputs = ecg_generator(real_ppg)
                if isinstance(outputs, tuple):
                    if len(outputs) == 3:
                        _, _, fake_ecg = outputs
                    elif len(outputs) == 4:
                        _, _, fake_ecg, _ = outputs
                    else:
                         fake_ecg = outputs[-1]
                else:
                    fake_ecg = outputs
            fake_ecg = smoother(fake_ecg, device)

        fake_ecg = torch.squeeze(fake_ecg).cpu().detach().numpy()
        real_ecg = torch.squeeze(real_ecg).cpu().detach().numpy()

        # Handle batch size 1
        if fake_ecg.ndim == 1:
            fake_ecg = np.expand_dims(fake_ecg, axis=0)
            real_ecg = np.expand_dims(real_ecg, axis=0)

        all_real_ecg.append(real_ecg)
        all_generated_ecg.append(fake_ecg)

    all_real_ecg = np.vstack(all_real_ecg)
    all_generated_ecg = np.vstack(all_generated_ecg)

    # Compute metrics in parallel
    # Note: n_jobs=1 to avoid issues in some environments, or keep 4
    eval_metrics_pairs = Parallel(n_jobs=4)(delayed(eval_metrics)(signal_a, signal_b) 
                                           for signal_a, signal_b in zip(all_generated_ecg, all_real_ecg))
    
    # Unzip the results
    rmse_list, p_list, r_squared_list, valid_list = zip(*eval_metrics_pairs)

    # Convert lists to numpy arrays for processing
    rmse_array = np.array(rmse_list)
    p_array = np.array(p_list)
    r_squared_array = np.array(r_squared_list)
    valid_array = np.array(valid_list)

    # Calculate mean and std for RMSE
    rmse_mean = np.mean(rmse_array)
    rmse_std = np.std(rmse_array)

    # Filter valid Pearson's r and R-squared
    valid_p = p_array[valid_array]
    valid_r_squared = r_squared_array[valid_array]
    zero_var_count = len(all_generated_ecg) - len(valid_p)

    if len(valid_p) > 0:
        p_mean = np.mean(valid_p)
        p_std = np.std(valid_p)
    else:
        p_mean = np.nan
        p_std = np.nan
        logging.warning("No valid Pearson's r values to compute mean and std.")

    if len(valid_r_squared) > 0:
        r_squared_mean = np.mean(valid_r_squared)
        r_squared_std = np.std(valid_r_squared)
    else:
        r_squared_mean = np.nan
        r_squared_std = np.nan
        logging.warning("No valid R-squared values to compute mean and std.")

    print('\nepoch:', steps)
    print(f'rmse_mean: {rmse_mean:.6f}, rmse_std: {rmse_std:.6f}')
    print(f'p_mean: {p_mean}, p_std: {p_std}')
    print(f'r_squared_mean: {r_squared_mean}, r_squared_std: {r_squared_std}')
    print(f'Zero Variance Count: {zero_var_count}')

    # Compute Frechet Distance
    ecg_fdists = []
    for i, sig in enumerate(all_generated_ecg):
        sig = np.expand_dims(sig, axis=0)
        real_ecg_sig = np.expand_dims(all_real_ecg[i], axis=0)
        ecg_fdists.append(similaritymeasures.frechet_dist(sig, real_ecg_sig))

    ecg_fdists = np.array(ecg_fdists)
    fdist_mean = np.mean(ecg_fdists)
    fdist_std = np.std(ecg_fdists)

    print(f'Frechet Distance Mean: {fdist_mean}, Frechet Distance Std: {fdist_std}')

    # Calculate PSNR and SSIM
    psnr_mean, psnr_std = PSNR(all_generated_ecg, all_real_ecg)
    ssim_mean, ssim_std = SSIM(all_generated_ecg, all_real_ecg)

    print(f'PSNR mean: {psnr_mean}, PSNR std: {psnr_std}')
    print(f'SSIM mean: {ssim_mean}, SSIM std: {ssim_std}')

    # Log metrics
    if writer:
        writer.add_scalars('Metrics', {
            'RMSE Mean': rmse_mean,
            'RMSE Std': rmse_std,
            'Pearson Coefficient Mean': p_mean,
            'Pearson Coefficient Std': p_std,
            'R-squared Mean': r_squared_mean,
            'R-squared Std': r_squared_std,
            'Frechet Distance Mean': fdist_mean,
            'Frechet Distance Std': fdist_std,
            'PSNR Mean': psnr_mean,
            'PSNR Std': psnr_std,
            'SSIM Mean': ssim_mean,
            'SSIM Std': ssim_std,
            'Zero Variance Count': zero_var_count
        }, steps)

    return rmse_mean, rmse_std, p_mean, p_std, r_squared_mean, r_squared_std, fdist_mean, fdist_std, psnr_mean, psnr_std, ssim_mean, ssim_std
