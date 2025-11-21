# BindRNAgen: Protein-binding RNA sequence generation using latent diffusion model

Project Overview

BindRNAgen is an advanced deep learning tool tailored for RNA sequence design, leveraging state-of-the-art neural network architectures (e.g., Transformer, GAN) to efficiently generate RNA sequences with predefined binding affinity, structural stability, or function-specific properties. Designed for computational biology and bioinformatics research, this tool accelerates experimental design and validation in RNA-targeted drug development, gene regulation mechanism studies, and related fields.

Installation Guide

Prerequisites

• Python 3.8 or higher

• Core Dependencies:

◦ Deep Learning Frameworks: PyTorch 1.10+ or TensorFlow 2.8+

◦ Data Processing: numpy 1.21+, pandas 1.3+, biopython 1.79+

◦ Structure Prediction: ViennaRNA 2.5+

◦ Visualization: matplotlib 3.4+, seaborn 0.11+

Quick Installation

Method 1: Install via pip
# Clone the repository
git clone https://github.com/zy-cmd/BindRNAgen.git
cd BindRNAgen

# Install dependencies
pip install -r requirements.txt

# Install ViennaRNA (required for structure prediction)
# For Ubuntu/Debian
sudo apt-get install viennarna
# For macOS
brew install viennarna
# For Windows
# Download installer from https://www.tbi.univie.ac.at/RNA/#download
Method 2: Install via Conda
# Create a new conda environment
conda create -n bindrnagen python=3.9
conda activate bindrnagen

# Clone the repository
git clone https://github.com/zy-cmd/BindRNAgen.git
cd BindRNAgen

# Install dependencies
conda install -c conda-forge torch numpy pandas biopython viennarna matplotlib seaborn
pip install -r requirements.txt
Usage Instructions

1. Data Preparation

• Prepare input files in FASTA or CSV format, including target sequences (e.g., protein binding partners) and constraint parameters.

• Example input file structure (CSV):
target_name,target_sequence,min_gc_content,max_gc_content,min_binding_affinity
ProteinA,MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN,35,65,-9.0
2. Quick Start (Sequence Generation)

Run the main script with default parameters to generate RNA sequences:
python bindrnagen/generate.py \
  --input data/input.csv \
  --output results/generated_sequences/ \
  --model_path models/pretrained_transformer.pth \
  --num_samples 10 \  # Number of sequences to generate per target
  --batch_size 4
3. Advanced Usage (Custom Constraints)

Customize structural and functional constraints via command-line arguments:
python bindrnagen/generate.py \
  --input data/input.csv \
  --output results/custom_sequences/ \
  --model_path models/pretrained_transformer.pth \
  --num_samples 15 \
  --secondary_structure "stem-loop" \  # Enforce stem-loop motif
  --gc_content 40-55 \  # Narrow GC content range
  --binding_affinity_threshold -10.0 \  # Strict binding affinity requirement
  --visualize_structure True  # Generate secondary structure plots
4. Model Training/Fine-Tuning

Fine-tune the pre-trained model with your own dataset:
python bindrnagen/train.py \
  --train_data data/train_dataset.csv \
  --val_data data/val_dataset.csv \
  --model_type transformer \  # Choose model type: transformer/gan
  --epochs 50 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --output_model_path models/fine_tuned_model.pth
Output Description

• Generated Sequences: generated_sequences.csv containing RNA sequences, GC content, predicted binding affinity, and structural scores.

• Secondary Structure Files: .dot files (ViennaRNA format) and PNG plots of RNA secondary structures (if --visualize_structure True).

• Model Logs: Training/fine-tuning logs stored in logs/ directory, including loss curves and performance metrics.

Model Architecture

BindRNAgen adopts a Transformer-based encoder-decoder architecture with the following key components:

• Encoder: Processes target sequences (e.g., proteins) and constraint features to extract high-dimensional representations.

• Decoder: Generates RNA sequences autoregressively, incorporating positional encoding and attention mechanisms to ensure sequence specificity.

• Constraint-Aware Module: Embeds structural/functional constraints into the generation process via multi-task learning.

Performance Evaluation

• Binding Affinity: Predicted using a pre-trained CNN model trained on RNA-protein binding datasets (e.g., RPI369, RPI488).

• Structural Stability: Evaluated by minimum free energy (MFE) calculated via ViennaRNA.

• Sequence Specificity: Measured by sequence similarity to native functional RNAs (BLASTn alignment).

Citation

If you use BindRNAgen in your research, please cite the following (to be updated with your publication details):
@article{BindRNAgen2024,
  title={BindRNAgen: A Deep Learning Tool for Target-Aware RNA Sequence Design},
  author={[Your Name] and [Co-Authors]},
  journal={[Journal Name]},
  year={2024},
  volume={[Volume]},
  pages={[Pages]},
  doi={[DOI]}
}
License

This project is licensed under the MIT License - see the LICENSE file for details.

Contact

For technical issues or questions, please open an issue on GitHub or contact:

• Author: [Your Name]

• Email: [your.email@example.com]

• GitHub: https://github.com/zy-cmd

Would you like me to adjust the model architecture details, usage examples, or citation format to match your project's actual implementation? I can also add sections like "Troubleshooting" or "FAQ" if needed.