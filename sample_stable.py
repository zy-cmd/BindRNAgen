# from unet import UNetV3
# from diffusion import Diffusion
# import torch
# import os
# import numpy as np
# from vae.model import VAE

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# model_dir_name = "/data/yzhou/rna_diffusion/checkpoints_train_stable_vae"
# model_path = os.path.join(model_dir_name, "qki_0928_v2_100_0.2_stable_diffusion.pt")
# print("Loading model from", model_path)
# print("Instantiating unet")

# unet = UNetV3(
#     dim=100,
#     channels=4,
#     dim_mults=(1, 2, 4),
#     resnet_block_groups=4,
# )

# print("Instantiating diffusion class")
# diffusion = Diffusion(
#     unet,
#     timesteps=100,
# )

# # Load checkpoint
# print("Loading checkpoint")
# checkpoint_dict = torch.load(model_path)
# diffusion.load_state_dict(checkpoint_dict["model"])

# # Send model to device
# print("Sending model to device")
# diffusion = diffusion.to("cuda")

# sample_bs = 32768
# nucleotides = ["A", "C", "G", "T"]

# print("Instantiating and loading VAE model...")
# vae = VAE(latent_channels=4)
# vae.load_state_dict(torch.load("/data/yzhou/rna_vae/total/1009_v2_1401.pth"))  
# vae = vae.to("cuda")
# vae.eval()

# # 指定包含 .pt 文件的路径
# pt_files_path = "/data/yzhou/rna_diffusion/prot_embedding_Esm2_qki"
# # 指定保存结果的目录
# output_dir = "/data/yzhou/rna_diffusion/generated_QKI_1012"

# def decode_one_hot_to_sequence(one_hot_tensor, alphabet):
#     """
#     此函数将模型输出的 one-hot 编码张量转换回序列字符串。
#     """
#     # 沿着通道维度（dim=0）找到最大值的索引
#     indices = torch.argmax(one_hot_tensor, dim=0)
    
#     # 将索引映射回对应的碱基字符
#     sequence = "".join([alphabet[i] for i in indices])
#     return sequence

# # 如果保存结果的目录不存在，则创建它
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 遍历指定路径下的所有 .pt 文件
# for root, dirs, files in os.walk(pt_files_path):
#     for file in files:
#         if file.endswith('.pt'):
#             file_path = os.path.join(root, file)
#             print(f"Processing file: {file_path}")
#             classes = torch.load(file_path).mean(dim=0).unsqueeze(0).repeat(sample_bs, 1)
#             classes = classes.to("cuda")
#             output_list = diffusion.sample_cross(
#                 classes=classes,
#                 shape=(sample_bs, 4, 8, 8),  # image size
#                 cond_weight=0,
#             )
            
#             final_latent = output_list[-1]

#             # 确保 final_latent 是一个 Tensor
#             # （如果 diffusion.sample_cross 返回的是 NumPy 数组，这行代码会将其转换）
#             if isinstance(final_latent, np.ndarray):
#                 final_latent = torch.from_numpy(final_latent).to("cuda")

#             reconstructed = vae.decoder(final_latent)
#             print("Reconstructed shape:", reconstructed.shape)
            
#             reconstructed = reconstructed.cpu() # 将结果移至 CPU 以便后续处理

#             generated_sequences = []
#             for i in range(sample_bs):
#                 seq_tensor = reconstructed[i]
#                 sequence_str = decode_one_hot_to_sequence(seq_tensor, nucleotides)
#                 generated_sequences.append(sequence_str)
            
#             # ===============================================================
#             # 新增：将生成的序列以 FASTA 格式保存到文件
#             # ===============================================================

#             # 1. 构建输出的 .fa 文件路径
#             # 获取不带扩展名的原始文件名，例如 "QKI_HepG2_ENCSR570WLM"
#             base_name = os.path.splitext(file)[0]
#             fa_file_name = f"{base_name}.fa"
#             fa_file_path = os.path.join(output_dir, fa_file_name)

#             # 2. 将生成的序列以 FASTA 格式写入文件
#             with open(fa_file_path, 'w') as f:
#                 for i, seq in enumerate(generated_sequences):
#                     # 写入 FASTA 标题行，例如 >QKI_HepG2_ENCSR570WLM_seq_1
#                     header = f">{base_name}_seq_{i+1}\n"
#                     f.write(header)
#                     # 写入序列本身
#                     f.write(seq + "\n")
            
#             print(f"Successfully saved {len(generated_sequences)} sequences to {fa_file_path}")
#             # ===============================================================

#             # except Exception as e:
#             #     print(f"Error processing file {file_path}: {e}")

# print(f"All sequences saved to {output_dir}")


from unet import UNetV3
from diffusion import Diffusion
import torch
import os
import numpy as np
from vae.model import VAE

# --- 可配置参数 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
model_dir_name = "/data/yzhou/rna_diffusion/checkpoints_train_stable_vae"
model_path = os.path.join(model_dir_name, "50_1014_exp.pt")
vae_model_path = "/data/yzhou/rna_vae/total/1013_v21_10_201.pth"
pt_files_path = "/data/yzhou/rna_diffusion/case" # 包含蛋白质嵌入.pt文件的路径
output_dir = "/data/yzhou/rna_diffusion/case_sequences"      # 保存生成序列的目录

# 10 v1 15 v2 01 v3 1 v4
# 【核心修改】在这里设置批次大小和总样本数
total_samples = 1000      # 您希望生成的总序列数量
generation_batch_size = 1000  # 每个小批次生成的序列数量，请根据您的GPU显存调整此数值

# --- 初始化模型 ---
print("Loading model from", model_path)
print("Instantiating unet")

unet = UNetV3(
    dim=100,
    channels=4,
    dim_mults=(1, 2, 4),
    resnet_block_groups=4,
)

print("Instantiating diffusion class")
diffusion = Diffusion(
    unet,
    timesteps=100,
)

# 加载扩散模型检查点
print("Loading checkpoint")
checkpoint_dict = torch.load(model_path)
diffusion.load_state_dict(checkpoint_dict["model"])
diffusion = diffusion.to("cuda")
diffusion.eval() # 设置为评估模式

print("Instantiating and loading VAE model...")
vae = VAE(latent_channels=4)
vae.load_state_dict(torch.load(vae_model_path))
vae = vae.to("cuda")
vae.eval() # 设置为评估模式

nucleotides = ["A", "C", "G", "T"]

def decode_one_hot_to_sequence(one_hot_tensor, alphabet):
    """
    此函数将模型输出的 one-hot 编码张量转换回序列字符串。
    """
    indices = torch.argmax(one_hot_tensor, dim=0)
    sequence = "".join([alphabet[i] for i in indices])
    return sequence

# 如果保存结果的目录不存在，则创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 主生成循环 ---
# 遍历指定路径下的所有 .pt 文件
for root, dirs, files in os.walk(pt_files_path):
    for file in files:
        if file.endswith('.pt'):
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")

            try:
                # 预加载蛋白质嵌入向量
                base_class = torch.load(file_path).mean(dim=0).unsqueeze(0).to("cuda")
                
                # 【核心修改】在这里实现分批次生成
                all_generated_sequences = []
                num_batches = (total_samples + generation_batch_size - 1) // generation_batch_size

                for i in range(num_batches):
                    print(f"  Generating batch {i+1}/{num_batches}...")

                    # 准备当前批次的条件向量
                    classes = base_class.repeat(generation_batch_size, 1)

                    # 1. 扩散模型采样，生成潜在向量
                    output_list = diffusion.sample_cross(
                        classes=classes,
                        shape=(generation_batch_size, 4, 8, 8),  # 使用较小的批次大小
                        cond_weight=0,
                    )
                    
                    final_latent = output_list[-1]
                    # final_latent = final_latent * 1.9586221485
                    if isinstance(final_latent, np.ndarray):
                        final_latent = torch.from_numpy(final_latent).to("cuda")

                    # 2. VAE解码器将潜在向量转换为 one-hot 编码
                    reconstructed = vae.decoder(final_latent)
                    reconstructed = reconstructed.cpu() # 将结果移至 CPU 以便后续处理

                    # 3. 将 one-hot 编码转换为序列字符串
                    for j in range(generation_batch_size):
                        seq_tensor = reconstructed[j]
                        sequence_str = decode_one_hot_to_sequence(seq_tensor, nucleotides)
                        all_generated_sequences.append(sequence_str)

                # --- 保存当前 .pt 文件对应的所有生成序列 ---
                base_name = os.path.splitext(file)[0]
                fa_file_name = f"{base_name}.fa"
                fa_file_path = os.path.join(output_dir, fa_file_name)

                with open(fa_file_path, 'w') as f:
                    # 只保存我们需要的 total_samples 数量的序列
                    for i, seq in enumerate(all_generated_sequences[:total_samples]):
                        header = f">{base_name}_seq_{i+1}\n"
                        f.write(header)
                        f.write(seq + "\n")
                
                print(f"Successfully saved {len(all_generated_sequences[:total_samples])} sequences to {fa_file_path}")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

print(f"\nAll sequences saved to {output_dir}")
