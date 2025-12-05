from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
plt.rcParams['figure.figsize'] = 17, 15
import os
import pandas as pd

import similaritymeasures

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False

def compute_gradient_penalty(D, real_samples, fake_samples, real_A, patch, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand((real_samples.size(0), 1, 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, real_A)
    fake = torch.full((real_samples.shape[0], *patch), 1, dtype=torch.float, device=device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


mean_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4,
                            bias=False, padding_mode='replicate', padding=1)
mean_conv.weight.data = torch.full_like(mean_conv.weight.data, 0.25)
pad = nn.ReplicationPad1d((0, 1))


def smoother(fake, device):
    fake = mean_conv.to(device)(fake)
    fake = pad.to(device)(fake)
    return fake


def sample_images(experiment_name, val_dataloader, generator, steps, device):
    """Saves generated signals from the validation set"""

    generator.eval()

    current_img_dir = "sample_signals/%s/%s.png" % (experiment_name, steps)

    signals = next(iter(val_dataloader))
    real_A_128 = signals[0].unsqueeze(dim=1).to(device).float()
    real_A_256 = signals[1].unsqueeze(dim=1).to(device).float()
    real_A = signals[2].unsqueeze(dim=1).to(device).float()
    real_B_128 = signals[3].unsqueeze(dim=1).to(device).float()
    real_B_256 = signals[4].unsqueeze(dim=1).to(device).float()
    real_B = signals[5].unsqueeze(dim=1).to(device).float()
    fake_B_128, fake_B_256, fake_B = generator(real_A)
    fake_B_128 = smoother(fake_B_128, device)
    fake_B_256 = smoother(fake_B_256, device)
    fake_B = smoother(fake_B, device)


    real_A_128 = torch.squeeze(real_A_128).cpu().detach().numpy()
    real_A_256 = torch.squeeze(real_A_256).cpu().detach().numpy()
    real_A = torch.squeeze(real_A).cpu().detach().numpy()
    real_B_128 = torch.squeeze(real_B_128).cpu().detach().numpy()
    real_B_256 = torch.squeeze(real_B_256).cpu().detach().numpy()
    real_B = torch.squeeze(real_B).cpu().detach().numpy()
    fake_B = torch.squeeze(fake_B).cpu().detach().numpy()
    fake_B_128 = torch.squeeze(fake_B_128).cpu().detach().numpy()
    fake_B_256 = torch.squeeze(fake_B_256).cpu().detach().numpy()

    fig, axes = plt.subplots(real_A.shape[0], 3)

    axes[0][0].set_title('Real PPG')
    axes[0][1].set_title('Real ECG')
    axes[0][2].set_title('Reconstructed ECG')

    for idx, signal in enumerate(real_A):
        axes[idx][0].plot(real_A[idx], color='cyan')
        axes[idx][1].plot(real_B[idx], color='maroon')
        axes[idx][2].plot(fake_B[idx], color='blue')

    fig.canvas.draw()
    fig.savefig(current_img_dir)
    plt.close(fig)

    return current_img_dir


def eval_metrics(signal_a, signal_b):
    rmse = np.sqrt(((signal_a - signal_b) ** 2).mean())
    p = stats.pearsonr(signal_a, signal_b)[0]
    r_squared = 1 - ((signal_a - signal_b) ** 2).sum() / ((signal_a - signal_a.mean()) ** 2).sum()
    return rmse, p, r_squared


def evaluate_generated_signal_quality(val_dataloader, ecg_generator, writer, steps, device):
    ecg_generator.eval()
    # ppg_generator.eval()

    all_real_ecg = []
    all_generated_ecg = []

    # all_real_ppg = []
    # all_generated_ppg = []

    for _, batch in enumerate(val_dataloader):
        real_A_128 = batch[0].unsqueeze(dim=1).to(device).float()
        real_A_256 = batch[1].unsqueeze(dim=1).to(device).float()
        real_A = batch[2].unsqueeze(dim=1).to(device).float()
        real_B_128 = batch[3].unsqueeze(dim=1).to(device).float()
        real_B_256 = batch[4].unsqueeze(dim=1).to(device).float()
        real_B = batch[5].unsqueeze(dim=1).to(device).float()
        

        fake_B = ecg_generator(real_A)[-1]
        fake_B = smoother(fake_B, device)
        fake_B = torch.squeeze(fake_B).cpu().detach().numpy()

        # fake_A = ppg_generator(torch.tensor(real_B).unsqueeze(dim=1).to(device).float())
        # fake_A = smoother(fake_A, device)
        # fake_A = torch.squeeze(fake_A).cpu().detach().numpy()

        all_real_ecg.append(real_B.squeeze().cpu().detach().numpy())
        all_generated_ecg.append(fake_B)

        # all_real_ppg.append(real_A.squeeze().cpu().detach().numpy())
        # all_generated_ppg.append(fake_A)

    all_real_ecg = np.vstack(all_real_ecg)
    all_generated_ecg = np.vstack(all_generated_ecg)

    # all_real_ppg = np.vstack(all_real_ppg)
    # all_generated_ppg = np.vstack(all_generated_ppg)



    eval_metrics_pairs = Parallel(n_jobs=4)(delayed(eval_metrics)(
        signal_a, signal_b) for signal_a, signal_b in zip(all_real_ecg, all_generated_ecg))
    res = list(zip(*eval_metrics_pairs))


    rmse_mean, rmse_std = np.mean(res[0]), np.std(res[0])
    p_mean, p_std = np.mean(res[1]), np.std(res[1])
    r_squared_mean, r_squared_std = np.mean(res[2]), np.std(res[2])

    print('\nepoch: ', steps)
    print('rmse_mean:', rmse_mean, ', rmse_std:', rmse_std)
    print('p_mean:', p_mean, ', p_std:', p_std)
    print('r_squared_mean:', r_squared_mean, ', r_squared_std:', r_squared_std)

    
    ecg_fdists = []

    for i, sig in enumerate(all_generated_ecg):
        sig = np.expand_dims(sig, axis=0)
        real_ecg = np.expand_dims(all_real_ecg[i], axis=0)
        ecg_fdists.append(similaritymeasures.frechet_dist(sig, real_ecg))

    ecg_fdists_mean, ecg_fdists_std = np.mean(ecg_fdists), np.std(ecg_fdists)
    
    print('frechet distance mean: {} frechet distance std: {}'.format(ecg_fdists_mean, ecg_fdists_std))

    if writer:
        writer.add_scalars('losses', {'rms_error': rmse_mean}, steps)
        writer.add_scalars('losses', {'p_mean': p_mean}, steps)
        writer.add_scalars('losses', {'r_squared_mean': r_squared_mean}, steps)
        writer.add_scalars('losses', {'frechet_distance': ecg_fdists_mean}, steps)

    return rmse_mean, rmse_std, p_mean, p_std, r_squared_mean, r_squared_std, ecg_fdists_mean, ecg_fdists_std

def plot_loss_curves(experiment_name):
    # Load losses CSV file
    losses_df = pd.read_csv(f'logs/{experiment_name}/losses.csv')
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses_df['epoch'], losses_df['g_loss'], label='Generator Loss')
    plt.plot(losses_df['epoch'], losses_df['d_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./logs/{experiment_name}/{experiment_name}_loss_curves.png')
    plt.close()

def plot_evaluation_metrics(experiment_name):
    # Load metrics CSV file
    metrics_df = pd.read_csv(f'logs/{experiment_name}/metrics.csv')
    
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['epoch'], metrics_df['rmse_mean'], label='RMSE')
    plt.plot(metrics_df['epoch'], metrics_df['P_coeff_mean'], label='P Coefficient')
    plt.plot(metrics_df['epoch'], metrics_df['r_squared_mean'], label='R Squared')
    plt.plot(metrics_df['epoch'], metrics_df['fdist_mean'], label='Frechet Distance')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Evaluation Metrics (Mean)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./logs/{experiment_name}/{experiment_name}_evaluation_metrics.png')
    plt.close()


# Initialize loss parameters
EPSILON = 1e-10
BP_LOW = 0.5
BP_HIGH = 50
BP_DELTA = 0.1

# Ideal Bandpass filter
def ideal_bandpass(freqs, psd, low_hz, high_hz, device):
    freqs = torch.tensor(freqs, dtype=torch.float32, device=device)
    psd = torch.tensor(psd, dtype=torch.float32, device=device).clone().detach()
    freq_idcs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
    freqs = freqs[freq_idcs]
    psd = psd.unsqueeze(0)[:, freq_idcs]
    return freqs, psd

# Normalize PSD
def normalize_psd(psd):
    return psd / torch.sum(psd, keepdim=True, dim=1) ## treat as probabilities


# Bandwidth loss
def _IPR_SSL(freqs, psd, low_hz, high_hz, device):
    freqs = torch.tensor(freqs, dtype=torch.float32, device=device).clone().detach()
    psd = torch.tensor(psd, dtype=torch.float32, device=device).clone().detach()
    use_freqs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
    use_freqs = use_freqs.to(psd.device)
    use_energy = torch.sum(psd[use_freqs], dim=0)
    zero_energy = torch.sum(psd[~use_freqs], dim=0)
    denom = use_energy + zero_energy + EPSILON
    ipr_loss = torch.mean(zero_energy / denom)
    return ipr_loss


def IPR_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, device=None):
    freqs = torch.tensor(freqs, dtype=torch.float32, device=device).clone().detach()
    if speed is None:
        ipr_loss = _IPR_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, device=device)
    else:
        batch_size = psd.shape[0]
        # ipr_losses = torch.ones((batch_size,1)).to(device)
        ipr_losses = torch.ones((batch_size, 1), device=device, dtype=torch.float32)
        for b in range(batch_size):
            low_hz_b = low_hz * speed[b]
            high_hz_b = high_hz * speed[b]
            psd_b = psd[b].view(1,-1)
            ipr_losses[b] = _IPR_SSL(freqs, psd_b, low_hz=low_hz_b, high_hz=high_hz_b, device=device)
        ipr_loss = torch.mean(ipr_losses)
    return ipr_loss

# Variance loss
def _EMD_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, normalized=False, bandpassed=False, device=None):
    ''' Squared earth mover's distance to uniform distribution.
    '''
    if not bandpassed:
        freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz, device)
    if not normalized:
        psd = normalize_psd(psd)
    B,T = psd.shape
    psd = torch.sum(psd, dim=0) / B
    expected = ((1/T) * torch.ones(T, device=device, dtype=torch.float32)) #uniform distribution
    emd_loss = torch.mean(torch.square(torch.cumsum(psd, dim=0) - torch.cumsum(expected, dim=0)))
    return emd_loss


def EMD_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, normalized=False, bandpassed=False, device=None):
    ''' Squared earth movers distance to uniform distribution.
    '''
    if speed is None:
        emd_loss = _EMD_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, normalized=normalized, bandpassed=bandpassed, device=device)
    else:
        B = psd.shape[0]
        expected = torch.zeros_like(freqs).to(device)
        for b in range(B):
            speed_b = speed[b]
            low_hz_b = low_hz * speed_b
            high_hz_b = high_hz * speed_b
            supp_idcs = torch.logical_and(freqs >= low_hz_b, freqs <= high_hz_b)
            uniform = torch.zeros_like(freqs)
            uniform[supp_idcs] = 1 / torch.sum(supp_idcs)
            expected = expected + uniform.to(device)
        lowest_hz = low_hz*torch.min(speed)
        highest_hz = high_hz*torch.max(speed)
        bpassed_freqs, psd = ideal_bandpass(freqs, psd, lowest_hz, highest_hz, device)
        bpassed_freqs, expected = ideal_bandpass(freqs, expected[None,:], lowest_hz, highest_hz, device)
        expected = expected[0] / torch.sum(expected[0]) #normalize expected psd
        psd = normalize_psd(psd) # treat all samples equally
        psd = torch.sum(psd, dim=0) / B # normalize batch psd
        emd_loss = torch.mean(torch.square(torch.cumsum(psd, dim=0) - torch.cumsum(expected, dim=0)))
    return emd_loss

# Sparsity loss
def _SNR_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, freq_delta=BP_DELTA, normalized=False, bandpassed=False, device=None):
    ''' We treat this as a dynamic IPR dependent on the maximum predicted frequency.
        Arguments:
            freq_delta (float): pad for maximum frequency window we integrate over in Hertz
    '''
    if not bandpassed:
        freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz,device)
    signal_freq_idx = torch.argmax(psd, dim=1)
    signal_freq_idx = signal_freq_idx.to(device)
    freqs = freqs.to(device)
    psd = psd.to(device)
    signal_freq = freqs[signal_freq_idx].view(-1, 1)
    freqs = freqs.repeat(psd.shape[0], 1)
    low_cut = signal_freq - freq_delta
    high_cut = signal_freq + freq_delta

    # Move low_cut and high_cut to the same device as freqs
    low_cut = low_cut.to(device)
    high_cut = high_cut.to(device)

    # Perform logical operation
    band_idcs = torch.logical_and(freqs >= low_cut, freqs <= high_cut)

    signal_band = torch.sum(psd * band_idcs, dim=1)
    noise_band = torch.sum(psd * ~band_idcs, dim=1)
    denom = signal_band + noise_band + EPSILON
    snr_loss = torch.mean(noise_band / denom)
    return snr_loss


def SNR_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, freq_delta=BP_DELTA, normalized=False, bandpassed=False, device=None):
    ''' We treat this as a dynamic IPR dependent on the maximum predicted frequency.
        Arguments:
            freq_delta (float): pad for maximum frequency window we integrate over in Hertz
    '''
    if speed is None:
        snr_loss = _SNR_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, freq_delta=freq_delta, normalized=normalized, bandpassed=bandpassed, device=device)
    else:
        batch_size = psd.shape[0]
        snr_losses = torch.ones((batch_size,1)).to(device)
        for b in range(batch_size):
            low_hz_b = low_hz * speed[b]
            high_hz_b = high_hz * speed[b]
            snr_losses[b] = _SNR_SSL(freqs, psd[b].view(1,-1), low_hz=low_hz_b, high_hz=high_hz_b, freq_delta=freq_delta, normalized=normalized, bandpassed=bandpassed, device=device)
        snr_loss = torch.mean(snr_losses)
    return snr_loss

#PTT loss



# Initialize loss parameters
l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()

# Guided Attention Loss Function
def guided_attention_loss(m_hat, m):
    L_attn = l1_loss(m_hat, m)
    return L_attn