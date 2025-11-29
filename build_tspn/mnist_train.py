import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from tools import AttrDict, Every
from mnist_set import MnistSet
from chamfer_distance import chamfer_distance_smoothed
import mnist_example
import datetime
import math
import argparse
import os
import re
import decimal


def set_config():
    config = AttrDict()
    # model config
    config.train_split = 80
    config.trans_layers = 3
    config.trans_attn_size = 256
    config.trans_num_heads = 4
    config.encoder_latent = 256
    config.encoder_output_channels = 64
    config.fspool_n_pieces = 20
    config.size_pred_width = 128
    config.train_steps = 100
    config.pad_value = -1
    config.tspn_learning_rate = 0.001
    config.prior_learning_rate = 0.1
    config.set_pred_learning_rate = 0.001
    config.weight_decay = 0.0001
    config.log_every = 500

    config.num_epochs = 100
    config.batch_size = 32
    return config


class TspnAutoencoder:
    def __init__(self, load_step, config, dataset):
        self.c = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.should_eval = Every(config.train_steps)
        self.should_log = Every(config.log_every)
        self.dataset = dataset

        self.max_set_size = dataset.max_num_elements
        self.element_size = dataset.element_size

        from tspn import TSPN  
        self.tspn = TSPN(config.encoder_latent, config.encoder_output_channels, config.fspool_n_pieces,
                         config.trans_layers, config.trans_attn_size, config.trans_num_heads,
                         dataset.element_size, config.size_pred_width, config.pad_value,
                         dataset.max_num_elements).to(self.device)

        self.tspn_optimiser = optim.Adam(self.tspn.get_autoencoder_weights(), lr=config.tspn_learning_rate)
        self.prior_optimiser = optim.Adam(self.tspn.get_prior_weights(), lr=config.prior_learning_rate)
        self.size_pred_optimiser = optim.Adam(self.tspn.get_size_predictor_weights(), lr=config.set_pred_learning_rate)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = f'logs/metrics/{current_time}'
        self.checkpoint_dir = f'logs/checkpoints/{current_time}'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(self.train_log_dir)

        if load_step is not None:
            self.load_checkpoint(load_step)

    def load_checkpoint(self, load_step):
        checkpoint_folder = "logs/checkpoints/"

        def extract_num(value):
            num = re.sub(r"\D", "", value)
            return decimal.Decimal(num) if num else decimal.Decimal(0)

        run_folders = sorted([f.path for f in os.scandir(checkpoint_folder) if f.is_dir()], key=extract_num)
        latest_run = run_folders[-1]
        step_ckpnts = sorted([f.path for f in os.scandir(latest_run) if f.is_dir()], key=extract_num)

        if load_step == -1:
            step_folder = step_ckpnts[-1]
        else:
            step_folder = [x for x in step_ckpnts if str(load_step) in x][0]

        self.tspn.load_state_dict(torch.load(os.path.join(step_folder, 'tspn.pt')))
        print(f"Loaded checkpoint from {step_folder}")

   
    def train_tspn(self):
        self.dataset.active_split = "train"
        train_loader = DataLoader(self.dataset, batch_size=self.c.batch_size, shuffle=True)
        self.dataset.active_split = "val"
        val_loader = DataLoader(self.dataset, batch_size=self.c.batch_size)

        print("Training prior...")
        for step, (images, sets, sizes, labels) in enumerate(train_loader):
            sets, sizes = sets.to(self.device), sizes.to(self.device)
            prior_loss = self.train_prior_step(sets, sizes)
            self.writer.add_scalar('train/prior_loss', prior_loss.item(), step)

        print("Training TSPN autoencoder...")
        global_step = 0
        for epoch in range(self.c.num_epochs):
            for batch_idx, (images, sets, sizes, labels) in enumerate(train_loader):
                sets, sizes = sets.to(self.device), sizes.to(self.device)
                model_loss = self.train_tspn_step(sets, sizes)
                global_step += 1
                self.writer.add_scalar('train/model_loss', model_loss.item(), global_step)

                if self.should_log(global_step):
                    print(f"Logging step {global_step}")
                    save_path = f"{self.checkpoint_dir}/{global_step}/"
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(self.tspn.state_dict(), os.path.join(save_path, "tspn.pt"))

            val_prior_loss_sum, val_model_loss_sum = 0.0, 0.0
            with torch.no_grad():
                for val_step, (images, sets, sizes, labels) in enumerate(val_loader):
                    sets, sizes = sets.to(self.device), sizes.to(self.device)
                    val_prior_loss, val_model_loss, _, _ = self.eval_tspn_step(sets, sizes)
                    val_prior_loss_sum += val_prior_loss.item()
                    val_model_loss_sum += val_model_loss.item()

            self.writer.add_scalar('val/prior_loss', val_prior_loss_sum / len(val_loader), global_step)
            self.writer.add_scalar('val/model_loss', val_model_loss_sum / len(val_loader), global_step)

   
    def train_size_predictor(self):
        train_loader = DataLoader(self.dataset.get_train_set(), batch_size=self.c.batch_size, shuffle=True)
        val_loader = DataLoader(self.dataset.get_val_set(), batch_size=self.c.batch_size)
        global_step = 0

        for epoch in range(self.c.num_epochs):
            print(f"Size predictor training epoch {epoch}")
            for (images, sets, sizes, labels) in train_loader:
                sets, sizes = sets.to(self.device), sizes.to(self.device)
                loss = self.train_size_predictor_step(sets, sizes)
                global_step += 1
                self.writer.add_scalar('train/size_predictor_loss', loss.item(), global_step)
            
            total_sq_err, total_count = 0, 0
            with torch.no_grad():
                for (images, sets, sizes, labels) in val_loader:
                    sets, sizes = sets.to(self.device), sizes.to(self.device)
                    pred_sizes, loss = self.eval_size_predictor_step(sets, sizes)
                    total_sq_err += ((pred_sizes - sizes) ** 2).sum().item()
                    total_count += sizes.numel()
            rmse = math.sqrt(total_sq_err / total_count)
            self.writer.add_scalar('val/size_predictor_RMSE', rmse, global_step)

    
    def prior_loss(self, initial_set, sizes):
        sampled_set = self.tspn.sample_prior(sizes)
        mask = (initial_set[:, :, 0] != self.c.pad_value)
        initial_flat = initial_set[mask]
        initial_flat = initial_flat.to(self.device).float().requires_grad_(True)
        sampled_set = sampled_set.to(self.device).float().requires_grad_(True)


        log_prob = self.tspn.prior_log_prob(initial_flat, sampled_set)
        prior_loss = -log_prob.mean()
        padded_samples = self.tspn.pad_samples(sampled_set, sizes, self.max_set_size, self.element_size, self.c.pad_value)
        return padded_samples, prior_loss

    def train_prior_step(self, x, sizes):
        self.prior_optimiser.zero_grad()
        _, loss = self.prior_loss(x, sizes)
        loss.backward()
        self.prior_optimiser.step()
        return loss

    def tspn_loss(self, x, sampled_set, sizes):
        pred_set = self.tspn(x, sampled_set, sizes)
        dist = chamfer_distance_smoothed(x, pred_set, sizes)
        return pred_set, dist.mean()

    def train_tspn_step(self, initial_set, sizes):
        sampled_set = self.tspn.sample_prior_batch(sizes)
        self.tspn_optimiser.zero_grad()
        _, loss = self.tspn_loss(initial_set, sampled_set, sizes)
        loss.backward()
        self.tspn_optimiser.step()
        return loss

    def eval_tspn_step(self, x, sizes):
        with torch.no_grad():
            padded_samples, prior_loss = self.prior_loss(x, sizes)
            pred_set, model_loss = self.tspn_loss(x, padded_samples, sizes)
        return prior_loss, model_loss, padded_samples, pred_set

    def size_predictor_loss(self, embedded_sets, sizes):
        pred_sizes = self.tspn.predict_size(embedded_sets)
        one_hot = torch.nn.functional.one_hot(sizes - 1, self.max_set_size).float()
        loss = torch.nn.functional.cross_entropy(pred_sizes, one_hot.argmax(dim=1))
        predicted_sizes = pred_sizes.argmax(dim=1) + 1
        return predicted_sizes, loss

    def train_size_predictor_step(self, initial_sets, sizes):
        self.size_pred_optimiser.zero_grad()
        embedded_sets = self.tspn.encode_set(initial_sets, sizes)
        _, loss = self.size_predictor_loss(embedded_sets, sizes)
        loss.backward()
        self.size_pred_optimiser.step()
        return loss

    def eval_size_predictor_step(self, initial_sets, sizes):
        with torch.no_grad():
            embedded_sets = self.tspn.encode_set(initial_sets, sizes)
            preds, loss = self.size_predictor_loss(embedded_sets, sizes)
        return preds, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--step', type=int)
    parser.add_argument('-p', '--predictor', action='store_true')
    args = parser.parse_args()

    config = set_config()
    dataset = MnistSet(config.train_split, config.pad_value, 20)
    trainer = TspnAutoencoder(args.step, config, dataset)

    if args.predictor:
        trainer.train_size_predictor()
    else:
        trainer.train_tspn()
