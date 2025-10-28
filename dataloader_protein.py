import os
import pickle
import random
from typing import Any
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from utils.utils import one_hot_encode

import torch
from torch.utils.data import Dataset, DataLoader
from utils.utils import one_hot_encode
nucleotides = ["A", "C", "G", "T"]


def read_sequences_from_txt(file_path):
    sequences = []
    file_names = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                parts = line.split('\t')
                if len(parts) == 2:
                    sequence, file_name = parts
                    sequences.append(sequence)
                    file_names.append(file_name)
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}。")
    return sequences, file_names


class SequenceDataset(Dataset):
    def __init__(self, encoded_sequences,protein_name):
        self.encoded_sequences = encoded_sequences
        self.protein_name = protein_name
    
    def __len__(self):
        return len(self.encoded_sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded_sequences[idx], dtype=torch.float32)
        x = torch.unsqueeze(x, dim=0)
        x = x.transpose(-1,-2)
        protein_name = self.protein_name[idx]
        path = os.path.join('/data/yzhou/rna_diffusion/prot_embedding_Esm2', protein_name)
        path_with_ext = path + '.pt'
        y = torch.load(path_with_ext).mean(dim=0)
        return x,y

class SequenceLucaDataset(Dataset):
    def __init__(self, encoded_sequences,protein_name):
        self.encoded_sequences = encoded_sequences
        self.protein_name = protein_name
    
    def __len__(self):
        return len(self.encoded_sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded_sequences[idx], dtype=torch.float32)
        # x = torch.unsqueeze(x, dim=0)
        # x = x.transpose(-1,-2)
        protein_name = self.protein_name[idx]
        path = os.path.join('/data/yzhou/rna_diffusion/prot_embedding_LucaoneV2', protein_name)
        path_with_ext = path + '.pt'
        y = torch.load(path_with_ext).mean(dim=0)
        return x,y

class SequenceStable(Dataset):
    def __init__(self, encoded_sequences,protein_name):
        self.encoded_sequences = encoded_sequences
        self.protein_name = protein_name
    
    def __len__(self):
        return len(self.encoded_sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded_sequences[idx], dtype=torch.float32).transpose(-1,-2)
        # x = torch.unsqueeze(x, dim=0)
        # x = x.transpose(-1,-2)
        protein_name = self.protein_name[idx]
        path = os.path.join('/data/yzhou/rna_diffusion/prot_embedding_Esm2', protein_name)
        path_with_ext = path + '.pt'
        y = torch.load(path_with_ext).mean(dim=0)
        return x,y

def load_data(txt_file_path):
    # 读取txt文件中的序列
    sequences,file_name = read_sequences_from_txt(txt_file_path)

    encoded_sequences = [one_hot_encode(seq, nucleotides, 100) for seq in sequences if "N" not in seq]

    return encoded_sequences, file_name



# dataset = SequenceDataset(loaded_encoded_sequences, loaded_file_names)
# batch_size = 32
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # # 示例：使用数据加载器
# for x,y in dataloader:
#     # 这里可以将 batch 输入到模型中进行训练或推理
#     print(f"Batch shape: {x.shape}, Labels shape: {y.shape}")
#     break
# # 读取txt文件中的序列
# txt_file_path = '/data/yzhou/rna_diffusion/test/extracted_sequences_tatal.txt'
# sequences,file_name = read_sequences_from_txt(txt_file_path)
# # # 对序列进行编码
# encoded_sequences = [one_hot_encode(seq, nucleotides, 100) for seq in sequences if "N" not in seq]
# # # 创建数据集实例
# dataset = SequenceDataset(encoded_sequences, file_name)

# # # 创建数据加载器
# batch_size = 32
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # # 示例：使用数据加载器
# for x,y in dataloader:
#     # 这里可以将 batch 输入到模型中进行训练或推理
#     print(f"Batch shape: {x.shape}, Labels shape: {y.shape}")
#     break