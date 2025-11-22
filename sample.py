import torch
import os
import argparse
import numpy as np
from unet import UNet
from diffusion import Diffusion
from vae.model import VAE 

def decode_one_hot_to_sequence(one_hot_tensor, alphabet):

    indices = torch.argmax(one_hot_tensor, dim=0)
    sequence = "".join([alphabet[i] for i in indices])
    return sequence

def generate_sequences(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={args.gpu_id})")

    print("--- 1. Initializing Models ---")
    
    print("Instantiating UNet model...")
    unet = UNet(
        dim=args.unet_dim,
        channels=args.unet_channels,
        dim_mults=args.unet_dim_mults,
        resnet_block_groups=args.unet_resnet_block_groups,
    )
    
    print("Instantiating Diffusion class...")
    diffusion = Diffusion(
        unet,
        timesteps=args.diffusion_timesteps,
    )

    print("Instantiating VAE model...")
    vae = VAE(latent_channels=args.vae_latent_channels)
    
    print("--- 2. Loading Checkpoints ---")
    
    diffusion_model_path = os.path.join(args.model_dir_name, args.model_checkpoint_name)
    print(f"Loading Diffusion checkpoint from {diffusion_model_path}")
    checkpoint_dict = torch.load(diffusion_model_path, map_location=device)
    diffusion.load_state_dict(checkpoint_dict["model"])
    diffusion = diffusion.to(device)
    diffusion.eval()

    print(f"Loading VAE checkpoint from {args.vae_model_path}")
    vae.load_state_dict(torch.load(args.vae_model_path, map_location=device))
    vae = vae.to(device)
    vae.eval()
    
    nucleotides = ["A", "C", "G", "T"]
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    print("--- 3. Starting Generation Loop ---")
    
    for root, dirs, files in os.walk(args.pt_files_path):
        for file in files:
            if file.endswith('.pt'):
                file_path = os.path.join(root, file)
                print(f"\nProcessing file: {file_path}")

                try:
                    base_class = torch.load(file_path).mean(dim=0).unsqueeze(0).to(device)
                    
                    all_generated_sequences = []
                    num_batches = (args.total_samples + args.generation_batch_size - 1) // args.generation_batch_size

                    for i in range(num_batches):
                        current_batch_size = min(args.generation_batch_size, args.total_samples - len(all_generated_sequences))
                        if current_batch_size == 0:
                            break
                            
                        print(f"  Generating batch {i+1}/{num_batches} (size: {current_batch_size})...")

                        classes = base_class.repeat(current_batch_size, 1)

                        with torch.no_grad():
                            output_list = diffusion.sample_cross(
                                classes=classes,
                                shape=(current_batch_size, args.vae_latent_channels, args.latent_size, args.latent_size),
                                cond_weight=args.cond_weight,
                            )
                        
                        final_latent = output_list[-1]
                        if isinstance(final_latent, np.ndarray):
                            final_latent = torch.from_numpy(final_latent).to(device)

                        with torch.no_grad():
                            reconstructed = vae.decoder(final_latent)
                        reconstructed = reconstructed.cpu() # 将结果移至 CPU 以便后续处理

                        for j in range(current_batch_size):
                            seq_tensor = reconstructed[j]
                            sequence_str = decode_one_hot_to_sequence(seq_tensor, nucleotides)
                            all_generated_sequences.append(sequence_str)

                    # --- 保存当前 .pt 文件对应的所有生成序列 ---
                    base_name = os.path.splitext(file)[0]
                    fa_file_name = f"{base_name}.fa"
                    fa_file_path = os.path.join(args.output_dir, fa_file_name)

                    # 确保只保存所需的 total_samples 数量
                    sequences_to_save = all_generated_sequences[:args.total_samples]
                    
                    with open(fa_file_path, 'w') as f:
                        for i, seq in enumerate(sequences_to_save):
                            header = f">{base_name}_seq_{i+1}\n"
                            f.write(header)
                            f.write(seq + "\n")
                    
                    print(f"Successfully saved {len(sequences_to_save)} sequences to {fa_file_path}")

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    print(f"\nAll sequences saved to {args.output_dir}")
    print("Generation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNA Sequence Generation using Conditional Diffusion and VAE.")

    # --- 路径和文件配置 ---
    parser.add_argument("--gpu_id", type=str, default="6", help="CUDA_VISIBLE_DEVICES setting.")
    parser.add_argument("--model_dir_name", type=str, default="/data/yzhou/rna_diffusion/checkpoints_train_stable_vae", help="Directory containing the diffusion model checkpoint.")
    parser.add_argument("--model_checkpoint_name", type=str, default="50_1014_exp.pt", help="Diffusion model checkpoint file name (inside model_dir_name).")
    parser.add_argument("--vae_model_path", type=str, default="/data/yzhou/rna_vae/total/1013_v21_10_201.pth", help="Path to the VAE model checkpoint.")
    parser.add_argument("--pt_files_path", type=str, default="/data/yzhou/rna_diffusion/case", help="Path to the directory containing protein embedding .pt files.")
    parser.add_argument("--output_dir", type=str, default="/data/yzhou/rna_diffusion/case_sequences", help="Directory to save the generated FASTA sequences.")

    parser.add_argument("--total_samples", type=int, default=1000, help="Total number of sequences to generate for each .pt file.")
    parser.add_argument("--generation_batch_size", type=int, default=1000, help="Batch size for sequence generation (adjust based on VRAM).")

    parser.add_argument("--unet_dim", type=int, default=100, help="Base dimension of the UNet.")
    parser.add_argument("--unet_channels", type=int, default=4, help="Input/Output channels of the UNet (for VAE latent space).")
    parser.add_argument("--unet_dim_mults", type=tuple, default=(1, 2, 4), help="Dimension multipliers for UNet layers.")
    parser.add_argument("--unet_resnet_block_groups", type=int, default=4, help="ResNet block groups in UNet.")
    
    parser.add_argument("--diffusion_timesteps", type=int, default=100, help="Number of timesteps in the diffusion model.")
    
    parser.add_argument("--vae_latent_channels", type=int, default=4, help="Latent channels in VAE (matches unet_channels).")
    parser.add_argument("--latent_size", type=int, default=8, help="Spatial size of the VAE latent vector (e.g., 8x8).")
    
    parser.add_argument("--cond_weight", type=float, default=0, help="Conditional guidance weight for sampling (0 means unconditional).")

    args = parser.parse_args()
    
    generate_sequences(args)