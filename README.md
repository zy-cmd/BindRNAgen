# BindRNAgen: Protein-binding RNA sequence generation using latent diffusion model

BindRNAgen is an advanced deep learning tool tailored for RNA sequence design, built on a diffusion model architecture. 
# Installation Guide

Prerequisites

• Python 3.8 or higher

• Core Dependencies:

◦ Deep Learning Frameworks: PyTorch 1.10+ 

◦ Data Processing: numpy 1.21+, pandas 1.3+, biopython 1.79+

◦ Structure Prediction: ViennaRNA 2.5+

◦ Visualization: matplotlib 3.4+, seaborn 0.11+

# Quick Start

1. Clone the repository
```
git clone https://github.com/zy-cmd/BindRNAgen.git
cd BindRNAgen
pip install -r requirements.txt
```
2. Data Preparation

Please download the data and model checkpoint from Google Drive link and then put all files into the current folder.

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
python train_vae.py
```
# Output Description

• Generated Sequences: generated_sequences.fa containing RNA sequences.
