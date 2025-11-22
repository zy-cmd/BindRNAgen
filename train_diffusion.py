import torch
import os 
import argparse
import pickle
import wandb

from accelerate import Accelerator
from diffusion.dataloader_protein import load_data 
from diffusion.diffusion import Diffusion
from diffusion.unet import UNet 
from train_util import TrainLoopStable 
from vae.model import VAE
def train(args):

    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    print("Initializing Accelerator...")
    accelerator = Accelerator(
        split_batches=True, 
        log_with=["wandb"], 
        mixed_precision=args.mixed_precision
    )

    print(f"Loading data from {args.data_path}...")
    try:
        with open(args.data_path, 'rb') as f:
            loaded_data = pickle.load(f)
            loaded_encoded_sequences = loaded_data['encoded_sequences']
            loaded_file_names = loaded_data['file_names']
        print(f"Loaded {len(loaded_encoded_sequences)} sequences.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    device = accelerator.device
    
    print("Instantiating UNet model...")
    unet = UNet(
        dim=args.unet_dim,
        channels=args.unet_channels,
        dim_mults=args.unet_dim_mults,
        resnet_block_groups=args.unet_resnet_block_groups,
    )

    print(f"Loading VAE model from {args.vae_model_path}...")
    vae = VAE(latent_channels=args.vae_latent_channels)
    try:
        vae.load_state_dict(torch.load(args.vae_model_path, map_location=device))
        vae.eval()
        vae = vae.to(device)
    except Exception as e:
        print(f"Error loading VAE model: {e}")
        return

    print(f"Instantiating Diffusion model with timesteps={args.diffusion_timesteps}...")
    diffusion = Diffusion(
        unet,
        timesteps=args.diffusion_timesteps,
    )
    
    print("Initializing TrainLoopStable...")
    
    trainer = TrainLoopStable(
        data=loaded_encoded_sequences,
        protein_name=loaded_file_names,
        model=diffusion,
        vae_model=vae,
        accelerator=accelerator, 
        epochs=args.epochs,
        log_step_show=args.log_step_show,
        sample_epoch=args.sample_epoch,
        save_epoch=args.save_epoch,
        model_name=args.model_name,
        image_size=args.image_size,
        num_sampling_to_compare_cells=args.num_sampling_to_compare_cells,
        batch_size=args.batch_size,
    )

    print("--- Starting Training Loop ---")
    trainer.train_loop()
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurable Training Script for RNA Diffusion Model.")

    parser.add_argument("--gpu_id", type=str, default="2", help="CUDA_VISIBLE_DEVICES setting.")
    
    parser.add_argument("--data_path", type=str, default="encoded_sequences_and_file_names_new.pkl", help="Path to the pickled data file.")
    parser.add_argument("--vae_model_path", type=str, default="/data/yzhou/rna_vae/total/1013_v21_10_201.pth", help="Path to the pre-trained VAE model checkpoint.")

    parser.add_argument("--wandb_project", type=str, default="rna_diffusion_new", help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default="total", help="Weights & Biases run name.")

    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--log_step_show", type=int, default=10, help="Log information every N steps.")
    parser.add_argument("--save_epoch", type=int, default=50, help="Save model checkpoint every N epochs.")
    parser.add_argument("--model_name", type=str, default="1014_exp", help="Base name for the saved model checkpoints.")
    

    parser.add_argument("--unet_dim", type=int, default=100, help="Base dimension of the UNet.")
    parser.add_argument("--unet_channels", type=int, default=4, help="Input/Output channels of the UNet (VAE latent channels).")
    parser.add_argument("--unet_dim_mults", type=tuple, default=(1, 2, 4), help="Dimension multipliers for UNet layers.")
    parser.add_argument("--unet_resnet_block_groups", type=int, default=4, help="ResNet block groups in UNet.")
    parser.add_argument("--diffusion_timesteps", type=int, default=100, help="Number of timesteps in the diffusion model.")
    
    parser.add_argument("--vae_latent_channels", type=int, default=4, help="Latent channels in VAE (must match unet_channels).")

    args = parser.parse_args()
    
    train(args)