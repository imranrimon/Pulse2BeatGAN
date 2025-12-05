import argparse
from src.training.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pulse2BeatGAN Training")
    parser.add_argument("--experiment_name", type=str, default="P2E_Refactored", help="name of the experiment")
    parser.add_argument("--dataset_prefix", type=str, default="Dataset/BIDMC", help="path to the dataset")
    parser.add_argument("--dataset", type=str, default="bidmc", choices=["bidmc", "dalia", "wesad", "capnobase", "mimic", "uqvitalsigns"], help="dataset to use")
    parser.add_argument("--test_dataset", type=str, default=None, choices=["bidmc", "dalia", "wesad", "capnobase", "mimic", "uqvitalsigns"], help="dataset to use for testing (optional)")
    parser.add_argument("--test_dataset_prefix", type=str, default=None, help="path to the test dataset (optional)")
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
    parser.add_argument("--model", type=str, default="swin_unet", choices=["swin_unet", "swin_unet_gab", "unet", "multires_unet"], help="model architecture to use")
    parser.add_argument("--limit", type=int, default=None, help="limit number of samples for testing")

    args = parser.parse_args()
    train(args)
