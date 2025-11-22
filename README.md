# BindRNAgen: Protein-binding RNA sequence generation using latent diffusion model

BindRNAgen is an advanced deep learning tool tailored for RNA sequence design, built on a diffusion model architecture. 

# Installation Guide

Prerequisites

• Python 3.13 or higher

• Core Dependencies:

◦ Deep Learning Frameworks: PyTorch 2.60+ 

◦ Data Processing: numpy 2.1.2+, pandas  2.2.3+, biopython 1.85+


# Quick Start

1. Clone the repository
```
git clone https://github.com/zy-cmd/BindRNAgen.git
cd BindRNAgen
conda env create -f environment.yml
```
2. Data Preparation

Please download model checkpoint from [Google Drive](https://drive.google.com/drive/folders/1Zpr_9DvyUMfwNhBHo27sUxhjCYPLk_gp) link and then put all files into the current folder.

3. Quick Start (Sequence Generation)

Run the main script with default parameters to generate RNA sequences:
```
python generate.py \
    --total_samples 2000 \
    --generation_batch_size 1000 \
    --gpu_id "6" \
    --output_dir "/path/to/my/new/output"
```
4. Model Training

Train the model with your own dataset:
```
python train_diffusion.py \
    --gpu_id "6" \
    --data_path "/path/to/encode_data" \
    --vae_model_path "/path/to/pretrained_vae_model_path" \
    --epochs 100 \
    --model_name "/path/model_name"

```
5. VAE Model Training

Pretrain the VAE model

```
python train_vae.py \
    --epochs 100 \
    --kl_beta 0.01 \
    --gpu_id "7" \
    --wandb_run_name "beta_0.01_run" \
    --data_path "/path/to/encode_data" \ 
    --final_model_save_path "path/to/saved_model_path" 
```
# Output Description

• Generated Sequences: generated_sequences.fa containing RNA sequences.

# License
This project is licensed under the  [CC-BY-NC-4.0](https://github.com/zy-cmd/BindRNAgen/blob/main/LICENSE).

