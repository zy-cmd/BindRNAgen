import copy
from typing import Any

import torch
import torchvision.transforms as T
from accelerate import Accelerator
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader_protein import SequenceDataset
from dataloader_protein import SequenceLucaDataset
from dataloader_protein import SequenceStable
from utils.utils import EMA


class TrainLoopStable:
    def __init__(
        self,
        data: dict[str, Any],
        protein_name,
        model: torch.nn.Module,
        vae_model: torch.nn.Module,
        accelerator: Accelerator,
        epochs: int = 10000,
        log_step_show: int = 50,
        sample_epoch: int = 500,
        save_epoch: int = 500,
        model_name: str = "model_path",
        image_size: int = 200,
        num_sampling_to_compare_cells: int = 1000,
        batch_size: int = 960,
    ):
        self.encode_data = data
        self.protein_name = protein_name
        self.model = model
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.accelerator = accelerator
        self.epochs = epochs
        self.log_step_show = log_step_show
        self.sample_epoch = sample_epoch
        self.save_epoch = save_epoch
        self.model_name = model_name
        self.image_size = image_size
        self.num_sampling_to_compare_cells = num_sampling_to_compare_cells

        self.vae = vae_model
        self.vae.requires_grad_(False)
        self.vae.eval()
        if self.accelerator.is_main_process:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        self.train_kl, self.test_kl, self.shuffle_kl = 1, 1, 1
        self.seq_similarity = 1

        self.start_epoch = 1

        seq_dataset = SequenceStable(self.encode_data,self.protein_name)
        self.train_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def train_loop(self):
        self.model, self.optimizer, self.train_dl = self.accelerator.prepare(self.model, self.optimizer, self.train_dl)
        self.vae.to(self.accelerator.device) # Move VAE to the correct device manually

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                "dnadiffusion",
                init_kwargs={"wandb": {"notes": "testing wandb accelerate script"}},
            )

        for epoch in tqdm(range(self.start_epoch, self.epochs + 1)):
            self.model.train()

            for step, batch in enumerate(self.train_dl):
                self.global_step = epoch * len(self.train_dl) + step

                loss = self.train_step(batch)

                if self.global_step % self.log_step_show == 0 and self.accelerator.is_main_process:
                    self.log_step(loss, epoch)


            if epoch % self.save_epoch == 0 and self.accelerator.is_main_process:
                self.save_model(epoch)

    def train_step(self, batch):
        x, y = batch
        with torch.no_grad():
            mu, log_var = self.vae.encoder(x)
            latents = mu
        with self.accelerator.autocast():
            loss = self.model(latents, y)

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.accelerator.wait_for_everyone()
        self.optimizer.step()

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.ema.step_ema(self.ema_model, self.accelerator.unwrap_model(self.model))

        self.accelerator.wait_for_everyone()
        return loss

    def log_step(self, loss, epoch):
        if self.accelerator.is_main_process:
            self.accelerator.log(
                {
                    "train": self.train_kl,
                    "test": self.test_kl,
                    "shuffle": self.shuffle_kl,
                    "loss": loss.mean().item(),
                    "epoch": epoch,
                    "seq_similarity": self.seq_similarity,
                },
                step=self.global_step,
            )

    def save_model(self, epoch):
        checkpoint_dict = {
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "ema_model": self.accelerator.get_state_dict(self.ema_model),
        }
        torch.save(
            checkpoint_dict,
            f"checkpoints_train_stable_vae/{epoch}_{self.model_name}.pt",
        )

    def load(self, path):
        checkpoint_dict = torch.load(path)
        self.model.load_state_dict(checkpoint_dict["model"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.start_epoch = checkpoint_dict["epoch"]

        if self.accelerator.is_main_process:
            self.ema_model.load_state_dict(checkpoint_dict["ema_model"])

        self.train_loop()
