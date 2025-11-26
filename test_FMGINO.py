
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
import numpy as np
import random
from einops import rearrange
from tqdm import tqdm
from neuralop.models import GINO

EPSILON = 1e-4

from models.UNET2D import Unet

# Model Class
class FMGINO(nn.Module):

    def __init__(self,
                 gino_model:nn.Module,
                 fm_model:nn.Module):
        super().__init__()
        self.gino = gino_model
        self.fm_model = fm_model

    def forward(self, batch):
        input_geom = batch["input_geom"]
        latent_queries = batch["latent_queries"]
        output_queries = batch["output_queries"]
        x = batch["x"]
        x_t = batch["x_t"].permute((0, 2, 1))
        t = batch["t"]

        # Forward pass through GINO
        lr_out = self.gino(
            input_geom,
            latent_queries,
            output_queries,
            x=x
        ).permute((0, 2, 1))

        return self.fm_model(torch.cat([lr_out, x_t], dim=1), t)
    
    @torch.no_grad
    def ode_solve(self, batch, num_steps:int=100, split_size:int=64, device:torch.device=torch.device("cpu")):
        input_geom = batch["input_geom"]
        latent_queries = batch["latent_queries"]
        output_queries = batch["output_queries"]
        x = batch["x"]

        time_steps = torch.linspace(0, 1.0, num_steps + 1).to(device)
        hr_out = torch.randn(size=(1, output_queries.shape[0], 1)).to(device)
        
        output_queries_list = output_queries.split(split_size=split_size, dim=0)
        lr_out_list = []
        for output_query in output_queries_list:
            lr_out = self.gino(
                input_geom.to(device),
                latent_queries.to(device),
                output_query.to(device),
                x=x.to(device)
            )
            lr_out_list.append(lr_out)
        fm_lr_input = torch.cat(lr_out_list, dim=1).permute((0,2,1))
        for i in range(num_steps):            
            hr_out = self.fm_model(torch.cat([fm_lr_input, hr_out.permute((0,2,1))], dim=1), time_steps[i,None])
            # hr_out = time_steps[i] * hr_out + (1.0 - time_steps[i]) * torch.randn(size=(1, output_queries.shape[0], 1)).to(device)
        return hr_out

# Dataset & Dataloaders
   
class MMGNViz_Baseline(Dataset):
    def __init__(self, 
        data: torch.Tensor,
        coords: torch.Tensor,
        input_idx_list: torch.Tensor,
        target_idx_list: torch.Tensor,
        max_val: float=25.0,
    ):
        self.data = data / max_val
        self.coords = coords
        self.input_idx_list = input_idx_list
        self.target_idx_list = target_idx_list

    def __len__(self):
        return len(self.target_idx_list)

    def __getitem__(self, idx):
        input_coords_idx = rearrange(self.input_idx_list[idx], 'x y c -> (x y) c')
        target_coords_idx = rearrange(self.target_idx_list[idx], 'x y c -> (x y) c')

        input_t, input_x, input_y = input_coords_idx.T
        target_t, target_x, target_y = target_coords_idx.T
        
        input_data = self.data[input_t, input_x, input_y]
        target_data = self.data[target_t, target_x, target_y]

        input_coords = self.coords[input_t, input_x, input_y]
        target_coords = self.coords[target_t, target_x, target_y]

        return {
            "sparse_observation": input_data,
            "sparse_coords": input_coords,
            "target_observation": target_data,
            "target_coords": target_coords,
            "sparse_observation_idx": input_coords_idx,
            "target_observation_idx": target_coords_idx,
            "idx": idx,
        }

# # Hyperparameters
# data_dir = "./MMGN_data/gst_data.npz"
# # NUM_EPOCHS = 500
# EPSILON = 1e-4
# x_points, y_points, t_points = 24, 36, 102
# xtarget_points, ytarget_points, ttarget_points = 192, 288, 102
# ###
# xc_start_idx = 4 # For GST
# xc_end_idx = xc_start_idx + (x_points - 1)*8

# yc_start_idx = 4 # For GST
# yc_end_idx = yc_start_idx + (y_points - 1)*8
# ###
# # n_sample = 16
# DENOISE_STEPS = 16
# NOISE_SAMPLES = 16

# DROPOUT_RATE = 0.25

# n_sample = 16 if DROPOUT_RATE > 0 else 1

import argparse

# Create the parser
parser = argparse.ArgumentParser(
    description="Training Polymer Property Predictor"
)

parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--num-epochs", type=int, default=500)
parser.add_argument("--x-points", type=int, required=True)
parser.add_argument("--y-points", type=int, required=True)
parser.add_argument("--dropout-rate", type=float, default=0.0)
parser.add_argument("--xtarget-points", type=int, required=True)
parser.add_argument("--ytarget-points", type=int, required=True)
parser.add_argument("--t-points", type=int, required=True)
parser.add_argument("--model-dir", type=str, required=True)
parser.add_argument("--result-save-dir", type=str, required=True)

parser.add_argument("--n-samples", type=int, default=32)
parser.add_argument("--denoise-steps", type=int, default=8)
parser.add_argument("--noise-samples", type=int, default=8)

# Parse the arguments from the command line
args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Hyperparameters
data_dir = args.data_dir #"./MMGN_data/gst_data.npz"
NUM_EPOCHS = args.num_epochs #500

x_points, y_points, t_points = args.x_points, args.y_points, args.t_points #24, 36, 102
xtarget_points, ytarget_points, ttarget_points = args.xtarget_points, args.ytarget_points, args.t_points #192, 288, 102

DROPOUT_RATE = args.dropout_rate #0.25

# n_sample = 16
DENOISE_STEPS = args.denoise_steps #16
NOISE_SAMPLES = args.noise_samples #16

n_sample = args.n_samples if DROPOUT_RATE > 0 else 1

# load data
dat = np.load(data_dir)
lat = torch.from_numpy(dat["lats"])
lon = torch.from_numpy(dat["lons"])
gst = torch.from_numpy(dat["temperature"])

# Input Coordinate Indexes
xc_idx = torch.linspace(0, gst.shape[1]-1, gst.shape[1], dtype=int)
yc_idx = torch.linspace(0, gst.shape[2]-1, gst.shape[2], dtype=int)
tc_idx = torch.linspace(0, gst.shape[0]-1, gst.shape[0], dtype=int)

t_grid_idx, x_grid_idx, y_grid_idx = torch.meshgrid([tc_idx,xc_idx,yc_idx], indexing='ij')
input_geom_idx = torch.stack([t_grid_idx, x_grid_idx, y_grid_idx]).permute((1, 2, 3, 0))

# Input Coordinates
xc = torch.linspace(-1.0+EPSILON, 1.0-EPSILON, gst.shape[1], dtype=torch.float)
yc = torch.linspace(-1.0+EPSILON, 1.0-EPSILON, gst.shape[2], dtype=torch.float)
tc = torch.linspace(-1.0+EPSILON, 1.0-EPSILON, gst.shape[0], dtype=torch.float)

t_grid, x_grid, y_grid = torch.meshgrid([tc,xc,yc], indexing='ij')
input_geom = torch.stack([t_grid, x_grid, y_grid]).permute((1, 2, 3, 0))

# LR DATA Indexes
xc_idx_input = torch.linspace(0, gst.shape[1]-1, x_points, dtype=int)
yc_idx_input = torch.linspace(0, gst.shape[2]-1, y_points, dtype=int)
tc_idx_input = tc_idx #torch.linspace(0, gst.shape[0]-1, t_points, dtype=int)

t_grid_idx_input, x_grid_idx_input, y_grid_idx_input = torch.meshgrid(
    [tc_idx_input,
     xc_idx_input,
     yc_idx_input], 
    indexing='ij'
)

input_geom_idx_input = torch.stack(
    [t_grid_idx_input, 
     x_grid_idx_input, 
     y_grid_idx_input]).permute((1, 2, 3, 0))

print(f"Input Geometry Shape: {input_geom_idx_input.shape}")

# HR DATA Indexes
xc_idx_target = torch.linspace(0, gst.shape[1]-1, xtarget_points, dtype=int)
yc_idx_target = torch.linspace(0, gst.shape[2]-1, ytarget_points, dtype=int)
tc_idx_target = tc_idx #torch.linspace(0, gst.shape[0]-1, ttarget_points, dtype=int)

t_grid_idx_target, x_grid_idx_target, y_grid_idx_target = torch.meshgrid(
    [tc_idx_target,
     xc_idx_target,
     yc_idx_target], 
    indexing='ij'
)

input_geom_idx_target = torch.stack(
    [t_grid_idx_target, 
     x_grid_idx_target, 
     y_grid_idx_target]).permute((1, 2, 3, 0))

print(f"Input Geometry Shape: {input_geom_idx_target.shape}")

# # Train Dataset
# train_dataset = MMGNViz_Baseline(
#     data=gst,
#     coords=input_geom,
#     input_idx_list=input_geom_idx_input,
#     target_idx_list=input_geom_idx_target,
# )

# train_loader =  DataLoader(
#     dataset=train_dataset, 
#     batch_size=2, 
#     num_workers=1, 
#     pin_memory=True, 
#     shuffle=True)

# Validation Dataset
val_dataset = MMGNViz_Baseline(
    data=gst,
    coords=input_geom,
    input_idx_list=input_geom_idx_input,
    target_idx_list=input_geom_idx_target,
)

val_loader =  DataLoader(
    dataset=val_dataset, 
    batch_size=1, 
    num_workers=1, 
    pin_memory=True, 
    shuffle=False)

latent_ts = torch.linspace(-1,1,11)
latent_xs = torch.linspace(-1,1,51)
latent_ys = torch.linspace(-1,1,51)
latent_geom = torch.stack(torch.meshgrid([latent_ts, latent_xs, latent_ys], indexing='ij'))
latent_geom = latent_geom.permute(1,2,3,0) 

device = torch.device("cuda")

# Create Model
fm_model = Unet(in_dim=2, dim=16).to(device)
gino_model = GINO(in_channels=1,
    out_channels=1,
    gno_radius=0.1,
    gno_coord_dim=3,
    fno_in_channels=3,
    fno_channel_mlp_dropout=DROPOUT_RATE,
    out_gno_tanh=True,
).to(device)

model_dir = os.path.join(
    args.model_dir, 
    f"seed-{seed}", 
    f"xy({x_points}-{y_points})->({xtarget_points}-{ytarget_points})",
    f"t-{t_points}",
    f"fm-gino",
)

# result_save_dir = "./GST_results"

result_save_dir = os.path.join(
    args.result_save_dir, 
    f"seed-{seed}", 
    f"xy({x_points}-{y_points})->({xtarget_points}-{ytarget_points})",
    f"t-{t_points}",
) 

os.makedirs(result_save_dir, exist_ok=True)

# print("===== Starting Evaluation =====")

# gino_model.load_state_dict(torch.load(os.path.join(model_dir, "gino_model.pt"), weights_only=False), strict=True)
# fm_model.load_state_dict(torch.load(os.path.join(model_dir, "fm_model.pt"), weights_only=False), strict=True)

gino_model_filename = "gino_model_mcd.pt" if DROPOUT_RATE > 0 else "gino_model.pt"
fm_model_filename = "fm_model_mcd.pt" if DROPOUT_RATE > 0 else "fm_model.pt"

gino_model_filepath = os.path.join(model_dir, gino_model_filename)
fm_model_filepath = os.path.join(model_dir, fm_model_filename)

gino_model.load_state_dict(torch.load(gino_model_filepath, weights_only=False), strict=True)
fm_model.load_state_dict(torch.load(fm_model_filepath, weights_only=False), strict=True)

print(f"Loaded GINO model from {gino_model_filepath} and FM model {fm_model_filepath}")

# model.eval()
predictions, targets = [], []
# gino_model.eval()
gino_model.train() if DROPOUT_RATE > 0 else gino_model.eval()
fm_model.eval()

print(f"DROPOUT RATE: {DROPOUT_RATE:03.2f} || Num of Weight Samples: {n_sample} || Num of Noise Samples: {NOISE_SAMPLES} || Num of Denoising Steps: {DENOISE_STEPS}")

print("===== Starting Evaluation =====")

with torch.no_grad():
    for batch in tqdm(val_loader):
        # Get batch data
        input_coords = batch["sparse_coords"]#[0]
        input_coords = rearrange(input_coords, 'b q c -> (b q) c')
        input_data = batch["sparse_observation"]
        input_data = rearrange(input_data, 'b q -> 1 (b q) 1')

        target_coords = batch["target_coords"]#[0]
        target_coords = rearrange(target_coords, 'b q c -> (b q) c')
        target_data = batch["target_observation"]
        # target_data = rearrange(target_data, 'b q -> 1 (b q) 1')
        target_data = rearrange(target_data, 'b (x y) -> b 1 x y', x=xtarget_points, y=ytarget_points)
        
        # lr_context = []
        # for _ in range(n_sample):
        #     lr_context_mcd = gino_model(
        #         input_coords.to(device),
        #         latent_geom.to(device),
        #         target_coords.to(device),
        #         x=input_data.to(device)
        #     )

        #     lr_context_mcd = rearrange(lr_context_mcd, '1 (b x y) 1 -> b 1 x y', x=xtarget_points, y=ytarget_points)

        #     lr_context.append(lr_context_mcd)

        # lr_context = torch.cat(lr_context, dim=0)

        lr_context = gino_model(
            input_coords.to(device),
            latent_geom.to(device),
            target_coords.to(device),
            x=input_data.repeat((n_sample, 1, 1)).to(device)
        )

        lr_context = rearrange(lr_context, 'b (x y) 1 -> b 1 x y', x=xtarget_points, y=ytarget_points, b=n_sample)

        time_steps = torch.linspace(0, 1.0, DENOISE_STEPS + 1).to(device)

        HR_OUTS = []
        
        for _ in range(NOISE_SAMPLES):
            hr_out = torch.randn_like(target_data).repeat((n_sample,1,1,1)).to(device)
            
            for i in range(DENOISE_STEPS):            
                hr_out = fm_model(torch.cat([lr_context, hr_out], dim=1), time_steps[i,None])

            HR_OUTS.append(hr_out)

        HR_OUTS = torch.cat(HR_OUTS, dim=1)
        
        predictions.append(HR_OUTS.cpu().numpy())
        targets.append(target_data.reshape((xtarget_points, ytarget_points)).numpy())

print("===== Saving Predictions =====")

pred_filename = 'FMGINO-PREDS-MCD-AUEU.npz' if DROPOUT_RATE > 0 else 'FMGINO-PREDS.npz'

np.savez(
    os.path.join(result_save_dir, pred_filename), 
    prediction=np.array(predictions), 
    target=np.array(targets),
)