import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
    logging.warning("TensorBoard not found. Logging disabled.")
from tqdm import tqdm

from src.data.preprocessing import create_peak_mask
from src.models.swin_unet import SwinTransformerSys
from src.models.swin_unet_gab import SwinTransformerSysGAB
from src.models.variants import MultiResUnet, GeneratorUNet
from src.models.discriminator import Discriminator
from src.data.dataset import (
    BIDMCDataset, 
    DALIADataset, 
    WESADDataset, 
    CapnobaseDataset, 
    MIMICAFibDataset, 
    UQVitalSignsDataset
)
from src.utils.losses import compute_gradient_penalty, QRSLoss, QRSEnhancedLoss, SNR_SSL, IPR_SSL, EMD_SSL, calculate_psd
from src.utils.metrics import evaluate_generated_signal_quality

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def get_dataset(dataset_name, data_dir, target_sr=128, segment_length=512, limit=None):
    dataset_name = dataset_name.lower()
    if dataset_name == 'bidmc':
        return BIDMCDataset(data_dir, target_sr, segment_length, limit=limit)
    elif dataset_name == 'dalia':
        return DALIADataset(data_dir, target_sr, segment_length, limit=limit)
    elif dataset_name == 'wesad':
        return WESADDataset(data_dir, target_sr, segment_length, limit=limit)
    elif dataset_name == 'capnobase':
        return CapnobaseDataset(data_dir, target_sr, segment_length, limit=limit)
    elif dataset_name == 'mimic':
        return MIMICAFibDataset(data_dir, target_sr, segment_length, limit=limit)
    elif dataset_name == 'uqvitalsigns':
        return UQVitalSignsDataset(data_dir, target_sr, segment_length, limit=limit)
    elif dataset_name == 'uqvitalsigns':
        return UQVitalSignsDataset(data_dir, target_sr, segment_length, limit=limit)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def train(args):
    logging.info(f"DEBUG: args.model={args.model}")
    # Create directories
    os.makedirs(f"logs/{args.experiment_name}", exist_ok=True)
    os.makedirs(f"saved_models/{args.experiment_name}", exist_ok=True)
    os.makedirs(f"sample_signals/{args.experiment_name}", exist_ok=True)

    # Tensorboard writer
    if SummaryWriter:
        writer = SummaryWriter(f"logs/{args.experiment_name}")
    else:
        writer = None

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Initialize models
    # Generator
    # Initialize models
    # Generator
    if args.model == "swin_unet":
        ecg_generator = SwinTransformerSys(
            img_size=[128, 256, 512],
            patch_size=4,
            in_chans=1,
            num_classes=1,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            depths_decoder=[1, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=4,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            final_upsample="expand_first"
        ).to(device)
    elif args.model == "swin_unet_gab":
        ecg_generator = SwinTransformerSysGAB(
            img_size=[128, 256, 512],
            patch_size=4,
            in_chans=1,
            num_classes=1,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            depths_decoder=[1, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=4,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            final_upsample="expand_first"
        ).to(device)
    elif args.model == "multires_unet":
        ecg_generator = MultiResUnet(input_channels=1, num_classes=1).to(device)
    elif args.model == "unet":
        ecg_generator = GeneratorUNet(input_channels=1, output_channels=1).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Discriminators
    discriminator = Discriminator(in_channels=1, kernel_size=[6, 4, 3, 2], stride=[4, 4, 2, 1], bias=False).to(device)
    discriminator_256 = Discriminator(in_channels=1, kernel_size=[6, 4, 3, 2], stride=[3, 3, 2, 1], bias=False).to(device)
    discriminator_128 = Discriminator(in_channels=1, kernel_size=[4, 3, 2, 1], stride=[3, 2, 2, 1], bias=False).to(device)

    # Initialize weights
    # ecg_generator.apply(weights_init_normal) # SwinTransformer likely has its own init
    discriminator.apply(weights_init_normal)
    discriminator_128.apply(weights_init_normal)
    discriminator_256.apply(weights_init_normal)

    # Optimizers
    optimizer_G = optim.Adam(ecg_generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D_128 = optim.Adam(discriminator_128.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D_256 = optim.Adam(discriminator_256.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixel = torch.nn.L1Loss()
    criterion_peaks = QRSLoss() # Or QRSEnhancedLoss

    # Data Loader
    logging.info(f"Initializing training dataset: {args.dataset} from {args.dataset_prefix}")
    train_full_dataset = get_dataset(args.dataset, args.dataset_prefix, limit=args.limit)
    
    if len(train_full_dataset) == 0:
        logging.error("Training dataset is empty. Exiting.")
        return

    if args.test_dataset:
        # Cross-dataset evaluation
        test_prefix = args.test_dataset_prefix if args.test_dataset_prefix else args.dataset_prefix
        logging.info(f"Initializing test dataset: {args.test_dataset} from {test_prefix}")
        val_full_dataset = get_dataset(args.test_dataset, test_prefix, limit=args.limit)
        
        if len(val_full_dataset) == 0:
            logging.error("Test dataset is empty. Exiting.")
            return
            
        train_dataset = train_full_dataset
        val_dataset = val_full_dataset
    else:
        # Standard split
        train_size = int(0.8 * len(train_full_dataset))
        val_size = len(train_full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)

    logging.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Training Loop
    prev_time = time.time()
    for epoch in range(args.epoch, args.n_epochs):
        for i, batch in enumerate(train_loader):
            # Unpack batch
            # batch[0]: ppg_128
            # batch[1]: ppg_256
            # batch[2]: ppg
            # batch[3]: ecg_128
            # batch[4]: ecg_256
            # batch[5]: ecg
            
            real_ppg_128 = batch[0].unsqueeze(1).to(device)
            real_ppg_256 = batch[1].unsqueeze(1).to(device)
            real_ppg = batch[2].unsqueeze(1).to(device)
            
            real_ecg_128 = batch[3].unsqueeze(1).to(device)
            real_ecg_256 = batch[4].unsqueeze(1).to(device)
            real_ecg = batch[5].unsqueeze(1).to(device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            optimizer_D_128.zero_grad()
            optimizer_D_256.zero_grad()

            # Generate a batch of images
            # Generate a batch of images
            if args.model == "swin_unet_gab":
                # Generate masks on the fly
                rpeaks_masks = []
                for sig in real_ecg.cpu().numpy():
                    mask = create_peak_mask(sig.flatten(), sampling_rate=128)
                    rpeaks_masks.append(mask)
                rpeaks_masks = torch.tensor(np.array(rpeaks_masks)).float().unsqueeze(1).to(device)
                
                opeaks_masks = torch.zeros_like(rpeaks_masks).to(device)
                
                fake_ecg_128, fake_ecg_256, fake_ecg, _ = ecg_generator(real_ppg, rpeaks_masks, opeaks_masks)
            elif args.model == "swin_unet":
                fake_ecg_128, fake_ecg_256, fake_ecg = ecg_generator(real_ppg)
            else:
                # For models that only return the final output (unet, multires_unet)
                fake_ecg = ecg_generator(real_ppg)
                # Downsample for multi-scale discriminator
                fake_ecg_256 = F.interpolate(fake_ecg, scale_factor=0.5, mode='linear', align_corners=False)
                fake_ecg_128 = F.interpolate(fake_ecg, scale_factor=0.25, mode='linear', align_corners=False)

            # Real loss
            pred_real = discriminator(real_ecg, real_ppg)
            pred_real_128 = discriminator_128(real_ecg_128, real_ppg_128)
            pred_real_256 = discriminator_256(real_ecg_256, real_ppg_256)
            
            loss_real = -torch.mean(pred_real)
            loss_real_128 = -torch.mean(pred_real_128)
            loss_real_256 = -torch.mean(pred_real_256)

            # Fake loss
            pred_fake = discriminator(fake_ecg.detach(), real_ppg)
            pred_fake_128 = discriminator_128(fake_ecg_128.detach(), real_ppg_128)
            pred_fake_256 = discriminator_256(fake_ecg_256.detach(), real_ppg_256)
            
            loss_fake = torch.mean(pred_fake)
            loss_fake_128 = torch.mean(pred_fake_128)
            loss_fake_256 = torch.mean(pred_fake_256)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_ecg.data, fake_ecg.data, real_ppg.data, (1, 14), device)
            gradient_penalty_128 = compute_gradient_penalty(discriminator_128, real_ecg_128.data, fake_ecg_128.data, real_ppg_128.data, (1, 10), device)
            gradient_penalty_256 = compute_gradient_penalty(discriminator_256, real_ecg_256.data, fake_ecg_256.data, real_ppg_256.data, (1, 12), device)

            # Total loss
            loss_D = loss_real + loss_fake + args.lambda_gp * gradient_penalty
            loss_D_128 = loss_real_128 + loss_fake_128 + args.lambda_gp * gradient_penalty_128
            loss_D_256 = loss_real_256 + loss_fake_256 + args.lambda_gp * gradient_penalty_256

            loss_D.backward()
            loss_D_128.backward()
            loss_D_256.backward()
            
            optimizer_D.step()
            optimizer_D_128.step()
            optimizer_D_256.step()

            # -----------------
            #  Train Generator
            # -----------------
            if i % args.ncritic == 0:
                optimizer_G.zero_grad()

                # Generate a batch of images
                # Generate a batch of images
                if args.model == "swin_unet_gab":
                    # Re-use masks generated above or regenerate (re-using is better but variables scope...)
                    # We need to regenerate or ensure scope. 
                    # Actually, we are in the same loop iteration, so rpeaks_masks is available if defined above.
                    # But wait, the previous block was 'Train Discriminator'.
                    # We should move mask generation to the top of the loop.
                    
                    # (Quick fix: regenerate for now, or assume variables persist in python loop scope - which they do)
                    # However, if we skipped D training (unlikely in standard loop but possible), we might crash.
                    # But D training is every step.
                    
                    fake_ecg_128, fake_ecg_256, fake_ecg, l1_losses = ecg_generator(real_ppg, rpeaks_masks, opeaks_masks, use_l1_loss=True)
                elif args.model == "swin_unet":
                    fake_ecg_128, fake_ecg_256, fake_ecg = ecg_generator(real_ppg)
                else:
                    fake_ecg = ecg_generator(real_ppg)
                    fake_ecg_256 = F.interpolate(fake_ecg, scale_factor=0.5, mode='linear', align_corners=False)
                    fake_ecg_128 = F.interpolate(fake_ecg, scale_factor=0.25, mode='linear', align_corners=False)

                # Pixel-wise loss
                loss_pixel = criterion_pixel(fake_ecg, real_ecg)

                # Adversarial loss
                pred_fake = discriminator(fake_ecg, real_ppg)
                pred_fake_128 = discriminator_128(fake_ecg_128, real_ppg_128)
                pred_fake_256 = discriminator_256(fake_ecg_256, real_ppg_256)
                
                loss_GAN = -torch.mean(pred_fake)
                loss_GAN_128 = -torch.mean(pred_fake_128)
                loss_GAN_256 = -torch.mean(pred_fake_256)

                # Frequency losses
                freqs, psd = calculate_psd(fake_ecg)
                
                # Bandwidth loss (IPR)
                loss_bandwidth = IPR_SSL(freqs, psd, device=device)
                
                # Sparsity loss (SNR)
                loss_sparsity = SNR_SSL(freqs, psd, device=device)
                
                # Variance loss (EMD)
                loss_variance = EMD_SSL(freqs, psd, device=device)
                
                # Weighted sum of losses (Weights can be tuned)
                loss_G = loss_GAN + loss_GAN_128 + loss_GAN_256 + 100 * loss_pixel + 0.1 * loss_bandwidth + 0.1 * loss_sparsity + 0.1 * loss_variance
                
                if args.model == "swin_unet_gab":
                    # Add GAB L1 losses
                    # l1_losses is a list of losses
                    gab_loss = sum(l1_losses)
                    loss_G += 0.5 * gab_loss # Default lambda from legacy code

                loss_G.backward()
                optimizer_G.step()

                # Logging
                batches_done = epoch * len(train_loader) + i
                batches_left = args.n_epochs * len(train_loader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / (i + 1 if i > 0 else 1))
                prev_time = time.time()

                if i % 10 == 0:
                    logging.info(
                        f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] ETA: {time_left}"
                    )
                    writer.add_scalar('Loss/D', loss_D.item(), batches_done)
                    writer.add_scalar('Loss/G', loss_G.item(), batches_done)

        # Evaluation
        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            torch.save(ecg_generator.state_dict(), f"saved_models/{args.experiment_name}/generator_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"saved_models/{args.experiment_name}/discriminator_{epoch}.pth")
            
            # Run evaluation
            evaluate_generated_signal_quality(val_loader, ecg_generator, writer, epoch, device)

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="P2E_Refactored", help="name of the experiment")
    parser.add_argument("--dataset_prefix", type=str, default="Dataset/BIDMC", help="path to the dataset")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="Loss weight for gradient penalty")
    parser.add_argument("--ncritic", type=int, default=5, help="number of iterations of the critic per generator iteration")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")

    args = parser.parse_args()
    train(args)
