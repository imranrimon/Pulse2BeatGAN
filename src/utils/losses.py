import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.functional import l1_loss

# Initialize loss parameters
EPSILON = 1e-10
BP_LOW = 0.5
BP_HIGH = 50
BP_DELTA = 0.1

def calculate_psd(x, fs=128):
    # x: (B, L) or (B, 1, L)
    if x.dim() == 3:
        x = x.squeeze(1)
    
    # FFT
    fft = torch.fft.rfft(x, dim=-1)
    # Power Spectral Density
    psd = torch.abs(fft) ** 2
    
    # Frequencies
    n = x.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1/fs).to(x.device)
    
    return freqs, psd

class QRSLoss(nn.Module):
    def __init__(self, beta=5):
        super(QRSLoss, self).__init__()
        self.beta = beta

    def forward(self, input, target, exp_rpeaks, device):
        return l1_loss(input * (1 + self.beta * exp_rpeaks), target * (1 + self.beta * exp_rpeaks))
    
class QRSEnhancedLoss(nn.Module):
    def __init__(self, beta=5, sigma=1):
        super(QRSEnhancedLoss, self).__init__()
        self.beta = beta
        self.sigma = sigma

    def forward(self, y_pred, y_true, r_peaks):
        batch_size, _, sequence_length = y_true.shape

        loss = 0.0
        for i in range(batch_size):
            r_peak = r_peaks[i]
            r_peaks_batch = r_peak[r_peak != 0]
            weight = torch.zeros(sequence_length, device=y_true.device)
            for r in r_peaks_batch:
                weight += torch.exp(-((torch.arange(sequence_length, device=y_true.device) - r) ** 2) / (2 * self.sigma ** 2))
            weight = 1 + self.beta * weight
            loss += torch.mean(weight * torch.abs(y_true[i] - y_pred[i]))

        return loss / batch_size

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

# Ideal Bandpass filter
def ideal_bandpass(freqs, psd, low_hz, high_hz, device):
    # freqs: (N,)
    # psd: (B, N)
    if not isinstance(freqs, torch.Tensor):
        freqs = torch.tensor(freqs, dtype=torch.float32, device=device)
    if not isinstance(psd, torch.Tensor):
        psd = torch.tensor(psd, dtype=torch.float32, device=device)
    
    freq_idcs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
    freqs = freqs[freq_idcs]
    psd = psd[:, freq_idcs]
    return freqs, psd

# Normalize PSD
def normalize_psd(psd):
    return psd / (torch.sum(psd, keepdim=True, dim=1) + EPSILON) ## treat as probabilities

# Bandwidth loss
def _IPR_SSL(freqs, psd, low_hz, high_hz, device):
    if not isinstance(freqs, torch.Tensor):
        freqs = torch.tensor(freqs, dtype=torch.float32, device=device)
    if not isinstance(psd, torch.Tensor):
        psd = torch.tensor(psd, dtype=torch.float32, device=device)
        
    use_freqs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
    use_freqs = use_freqs.to(psd.device)
    
    # psd is (B, N)
    # We want to sum energy in the band vs out of band
    use_energy = torch.sum(psd[:, use_freqs], dim=1)
    zero_energy = torch.sum(psd[:, ~use_freqs], dim=1)
    denom = use_energy + zero_energy + EPSILON
    ipr_loss = torch.mean(zero_energy / denom)
    return ipr_loss


def IPR_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, device=None):
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
    
    B, T = psd.shape
    # Expected uniform distribution
    expected = ((1/T) * torch.ones(T, device=device, dtype=torch.float32)) 
    # Cumulative sum along frequency axis (dim 1)
    cdf_psd = torch.cumsum(psd, dim=1)
    cdf_expected = torch.cumsum(expected, dim=0).unsqueeze(0).expand(B, -1)
    
    emd_loss = torch.mean(torch.square(cdf_psd - cdf_expected))
    return emd_loss


def EMD_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, normalized=False, bandpassed=False, device=None):
    ''' Squared earth movers distance to uniform distribution.
    '''
    if speed is None:
        emd_loss = _EMD_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, normalized=normalized, bandpassed=bandpassed, device=device)
    else:
        # Simplified speed handling for now, assuming similar logic or just not used in this context yet
        # But keeping original logic structure if possible, but fixing batching
        # The original logic for speed was iterating batch.
        B = psd.shape[0]
        expected = torch.zeros_like(freqs).to(device)
        # ... (This part is complex to vectorize without more changes, let's assume speed is None for now as we don't use it in train.py)
        # If speed is used, we fall back to loop or need rewrite.
        # For now, just fixing the non-speed path which is what we use.
        return _EMD_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, normalized=normalized, bandpassed=bandpassed, device=device)
    return emd_loss

# Sparsity loss
def _SNR_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, freq_delta=BP_DELTA, normalized=False, bandpassed=False, device=None):
    ''' We treat this as a dynamic IPR dependent on the maximum predicted frequency.
        Arguments:
            freq_delta (float): pad for maximum frequency window we integrate over in Hertz
    '''
    if not bandpassed:
        freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz, device)
    
    # psd is (B, N)
    signal_freq_idx = torch.argmax(psd, dim=1) # (B,)
    
    if not isinstance(freqs, torch.Tensor):
        freqs = torch.tensor(freqs, dtype=torch.float32, device=device)
        
    signal_freq = freqs[signal_freq_idx].view(-1, 1) # (B, 1)
    
    # Broadcast freqs to (B, N)
    freqs_broadcast = freqs.unsqueeze(0).expand(psd.shape[0], -1)
    
    low_cut = signal_freq - freq_delta
    high_cut = signal_freq + freq_delta

    # Perform logical operation
    band_idcs = torch.logical_and(freqs_broadcast >= low_cut, freqs_broadcast <= high_cut)

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
