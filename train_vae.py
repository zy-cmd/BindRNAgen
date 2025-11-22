import pickle
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm
import wandb
import os
from sklearn.model_selection import train_test_split

from vae.model import VAE 

class RNADataset(Dataset):

    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx]).transpose(0, 1)
        return sequence


def train_vae(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={args.gpu_id})")

    wandb.init(
        project=args.wandb_project, 
        config=vars(args), # 记录所有 argparse 参数
        name=args.wandb_run_name
    )

    print(f"Loading and preparing data from {args.data_path}...")
    try:
        with open(args.data_path, 'rb') as f:
            loaded_data = pickle.load(f)
            loaded_encoded_sequences = loaded_data['encoded_sequences']
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("Splitting data into training and validation sets...")
    train_sequences, val_sequences = train_test_split(
        loaded_encoded_sequences,
        test_size=args.val_split_ratio,
        random_state=args.random_seed
    )

    train_dataset = RNADataset(train_sequences)
    val_dataset = RNADataset(val_sequences)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Data ready. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    model = VAE(latent_channels=args.latent_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    wandb.watch(model, log="all", log_freq=100)
    
    if not os.path.exists(args.periodic_save_dir):
        os.makedirs(args.periodic_save_dir)
        print(f"Created periodic save directory: {args.periodic_save_dir}")

    print("Starting training... 🚀")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss, total_recon_loss, total_kl_div = 0.0, 0.0, 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
        for batch in progress_bar:
            x = batch.to(device)
            optimizer.zero_grad()
            
            recon_x, mu, log_var = model(x)
            recon_loss, kl_div, loss = model.loss_function(recon_x, x, mu, log_var, args.kl_beta)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Log batch-level metrics
            wandb.log({
                "batch_train_loss": loss.item(),
                "batch_recon_loss": recon_loss.item(),
                "batch_kl_div": kl_div.item()
            })
            total_train_loss += loss.item() * len(batch)
            total_recon_loss += recon_loss.item() * len(batch)
            total_kl_div += kl_div.item() * len(batch)
        
        if (epoch + 1) % args.save_freq_epochs == 0:
            save_filename = f"{args.periodic_save_prefix}_{epoch+1}.pth"
            save_path = os.path.join(args.periodic_save_dir, save_filename)
            torch.save(model.state_dict(), save_path)
            print(f"\nCheckpoint saved to {save_path} at Epoch {epoch+1}")
            
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                x = batch.to(device)
                recon_x, mu, log_var = model(x)
                _, _, val_loss = model.loss_function(recon_x, x, mu, log_var, args.kl_beta)
                total_val_loss += val_loss.item() * len(batch)
        
        avg_train_loss = total_train_loss / len(train_dataset)
        avg_val_loss = total_val_loss / len(val_dataset)
        avg_recon_loss = total_recon_loss / len(train_dataset)
        avg_kl_div = total_kl_div / len(train_dataset)

        print(f"Epoch {epoch+1}/{args.epochs} -> Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
        
        # Log epoch-level metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "avg_recon_loss": avg_recon_loss,
            "avg_kl_div": avg_kl_div,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    # --- 保存最终模型 ---
    if args.final_model_save_path:
        print(f"Training finished. Saving final model to {args.final_model_save_path} 💾")
        torch.save(model.state_dict(), args.final_model_save_path)
        
        # Log final model as a wandb artifact
        artifact = wandb.Artifact('rna-vae-model-final', type='model')
        artifact.add_file(args.final_model_save_path)
        wandb.log_artifact(artifact)
    
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configurable VAE Training Script for RNA Sequences.")

    parser.add_argument("--gpu_id", type=str, default="6", help="CUDA_VISIBLE_DEVICES setting (e.g., '6' or '0,1').")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for data splitting.")
    
    parser.add_argument("--data_path", type=str, default="/data/yzhou/rna_vae/encoded_sequences_and_file_names_new.pkl", help="Path to the pickled data file.")
    parser.add_argument("--wandb_project", type=str, default="rna-vae-project-refactored", help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default="total_run_v1", help="Weights & Biases run name.")
    
    parser.add_argument("--latent_channels", type=int, default=4, help="Number of channels in the VAE latent space.")

    parser.add_argument("--epochs", type=int, default=200, help="Total number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=4096, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--val_split_ratio", type=float, default=0.01, help="Ratio of data to use for the validation set.")
    parser.add_argument("--kl_beta", type=float, default=0.001, help="Weight for the KL divergence term (beta).")

    parser.add_argument("--final_model_save_path", type=str, default="/data/yzhou/rna_vae/qki_only/qki_0928_v1.pth", help="Path to save the final model checkpoint.")
    parser.add_argument("--periodic_save_dir", type=str, default="/data/yzhou/rna_vae/total", help="Directory for periodic model saves (e.g., every N epochs).")
    parser.add_argument("--periodic_save_prefix", type=str, default="1013_v21_01", help="Filename prefix for periodic checkpoints (will append epoch number).")
    parser.add_argument("--save_freq_epochs", type=int, default=50, help="Frequency (in epochs) to save periodic checkpoints.")

    args = parser.parse_args()
    
    train_vae(args)