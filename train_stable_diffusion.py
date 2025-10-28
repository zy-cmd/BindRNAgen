from accelerate import Accelerator
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import wandb
from dataloader_protein import load_data
from diffusion import Diffusion
from unet import UNetV3
from train_util import TrainLoopStable
from vae.model import VAE
import torch

def train():
    wandb.init(project="rna_diffusion_new", name="total")

    accelerator = Accelerator(split_batches=True, log_with=["wandb"], mixed_precision="bf16")

    import pickle

    # 读取保存的 encoded_sequences 和 file_names
    with open('encoded_sequences_and_file_names_new.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
        loaded_encoded_sequences = loaded_data['encoded_sequences']
        loaded_file_names = loaded_data['file_names']

    # data,protein_name = load_data(
    #     "/data/yzhou/rna_diffusion/test/extracted_sequences_tatal.txt",
    # )

    unet = UNetV3(
        dim=100,
        channels=4,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4,
    )

    vae = VAE(latent_channels=4)
    # vae.load_state_dict(torch.load("/data/yzhou/rna_vae/total/0926_v2_201.pth"))
    vae.load_state_dict(torch.load("/data/yzhou/rna_vae/total/1013_v21_10_201.pth"))  
    # vae.load_state_dict(torch.load("/data/yzhou/rna_vae/total/1012_v3_0.15_51.pth"))  
    # vae.load_state_dict(torch.load("/data/yzhou/rna_vae/total/1009_v2_1401.pth"))   这个是0.2
    # vae.load_state_dict(torch.load("/data/yzhou/rna_vae/all_ckpt/0925_v4_0.2_checkpoint_epoch_181.pth")) 0.06
    vae.eval()

    diffusion = Diffusion(
        unet,
        timesteps=100,
    )

    TrainLoopStable(
        data=loaded_encoded_sequences,
        protein_name = loaded_file_names,
        model=diffusion,
        vae_model=vae,
        accelerator=accelerator, 
        epochs=100,
        log_step_show=10,
        sample_epoch=500,
        save_epoch=50,
        model_name="1014_exp",
        image_size=100,
        num_sampling_to_compare_cells=1000,
        batch_size=1024,
    ).train_loop()


if __name__ == "__main__":
    train()
