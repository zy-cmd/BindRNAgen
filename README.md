# BindRNAgen: Protein-binding RNA sequence generation using latent diffusion model

BindRNAgen is an advanced deep learning tool tailored for RNA sequence design, built on a diffusion model architecture. 
Installation Guide

Prerequisites

• Python 3.8 or higher

• Core Dependencies:

◦ Deep Learning Frameworks: PyTorch 1.10+ 

◦ Data Processing: numpy 1.21+, pandas 1.3+, biopython 1.79+

◦ Structure Prediction: ViennaRNA 2.5+

◦ Visualization: matplotlib 3.4+, seaborn 0.11+

Quick Installation

# Clone the repository
```
git clone https://github.com/zy-cmd/BindRNAgen.git
cd BindRNAgen
pip install -r requirements.txt
```
1. Data Preparation

Please download the data and model checkpoint from Google Drive link and then put all files into the current folder.

2. Quick Start (Sequence Generation)

Run the main script with default parameters to generate RNA sequences:
```
python bindrnagen/generate.py \
  --input data/input.csv \
  --output results/generated_sequences/ \
  --model_path models/pretrained_transformer.pth \
  --num_samples 10 \  # Number of sequences to generate per target
  --batch_size 4
```
3. Model Training/Fine-Tuning

Fine-tune the pre-trained model with your own dataset:
```
python bindrnagen/train.py \
  --train_data data/train_dataset.csv \
  --val_data data/val_dataset.csv \
  --model_type transformer \  # Choose model type: transformer/gan
  --epochs 50 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --output_model_path models/fine_tuned_model.pth
```

Output Description

• Generated Sequences: generated_sequences.csv containing RNA sequences, GC content, predicted binding affinity, and structural scores.

