
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
class MMGN_Baseline(Dataset):
    def __init__(self, 
        data: torch.Tensor,
        coords: torch.Tensor,
        input_idx_list: torch.Tensor,
        target_idx_list: torch.Tensor,
        max_val: float=25.0,
        input_points: int=128,
        target_points: int=1024,
    ):
        self.data = data / max_val
        self.coords = coords
        self.input_idx_list = input_idx_list
        self.target_idx_list = target_idx_list
        self.input_points = input_points
        self.target_points = target_points

    def __len__(self):
        return len(self.target_idx_list)

    def __getitem__(self, idx):
        input_coords_idx = rearrange(self.input_idx_list[idx], 'x y c -> (x y) c')
        target_coords_idx = rearrange(self.target_idx_list[idx], 'x y c -> (x y) c')

        input_coords_idx = input_coords_idx[random.sample(range(input_coords_idx.shape[0]), self.input_points)]
        target_coords_idx = target_coords_idx[random.sample(range(target_coords_idx.shape[0]), self.target_points)]

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
tc_idx_input = torch.linspace(0, gst.shape[0]-1, t_points, dtype=int)

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
tc_idx_target = torch.linspace(0, gst.shape[0]-1, ttarget_points, dtype=int)

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

# Train Dataset
train_dataset = MMGNViz_Baseline(
    data=gst,
    coords=input_geom,
    input_idx_list=input_geom_idx_input,
    target_idx_list=input_geom_idx_target,
)

train_loader =  DataLoader(
    dataset=train_dataset, 
    batch_size=1, 
    num_workers=1, 
    pin_memory=True, 
    shuffle=True)

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
# fm_model = Unet(in_dim=2, dim=16).to(device)
fm_model = Unet(in_dim=1, dim=16).to(device)
# gino_model = GINO(in_channels=1,
#     out_channels=1,
#     gno_radius=0.1,
#     gno_coord_dim=3,
#     fno_in_channels=3,
#     fno_channel_mlp_dropout=DROPOUT_RATE,
#     out_gno_tanh=True,
# ).to(device)
gino_model = GINO(
    in_channels=1,
    out_channels=1,
    projection_channels=8,
    gno_radius=0.1,
    fno_hidden_channels=8,
    fno_n_layers=1,
    gno_embed_channels=8,
    gno_embed_max_positions=10000,
    in_gno_channel_mlp_hidden_layers=[8,8,8],
    out_gno_channel_mlp_hidden_layers=[32,16],
    out_gno_tanh=True,
    fno_channel_mlp_dropout=DROPOUT_RATE,
).to(device)

params_list = [
    {'params': gino_model.parameters()},
    {'params': fm_model.parameters()}
]

optimizer = torch.optim.AdamW(params_list, lr=1e-3, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CyclicLR(
#     optimizer, 
#     base_lr=1e-4, 
#     max_lr=5e-4, 
#     step_size_up=100, 
#     step_size_down=None, 
#     mode='triangular', 
#     gamma=1.0,
#     cycle_momentum=False)
loss_fn = torch.nn.MSELoss(reduction='mean')

model_dir = os.path.join(
    args.model_dir, 
    f"seed-{seed}", 
    f"xy({x_points}-{y_points})->({xtarget_points}-{ytarget_points})",
    f"t-{t_points}",
    f"fm-gino",
) 

os.makedirs(model_dir, exist_ok=True)

print("===== Starting Training =====")

pbar = tqdm(range(NUM_EPOCHS))

for epoch in pbar:

    gino_model.train()
    fm_model.train()

    for batch in train_loader:
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

        # sample time (user's responsibility)
        x_1 = target_data.to(device)
        x_0 = torch.randn_like(x_1)#.to(device)
        t = torch.rand(x_1.shape[0]).to(device) 

        # x_t = t[:,None,None,None] * x_1 + (1.0 - t[:,None,None,None]) * x_0

        lr_context = gino_model(
            input_coords.to(device),
            latent_geom.to(device),
            target_coords.to(device),
            x=input_data.to(device)
        )

        lr_context = rearrange(lr_context, '1 (b x y) 1 -> b 1 x y', x=xtarget_points, y=ytarget_points)

        x_t = t[:,None,None,None] * lr_context + (1.0 - t[:,None,None,None]) * x_0
        
        # lr_input = torch.cat([lr_context, x_t], dim=1)

        # output = fm_model(lr_input, t)
        output = fm_model(x_t, t)

        # Calculate loss
        # loss = loss_fn(output, dx_t.to(device))
        loss = loss_fn(output, x_1)

        # Update model weights
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gino_model.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(fm_model.parameters(), 10.0)
        optimizer.step() 

        # Print loss
        pbar.set_description(f"Epoch: {epoch+1:04d}/{NUM_EPOCHS:04d} || Loss: {loss.item():08.8f}")
    
    # # Update learning rate
    # scheduler.step()
    
    # Save model weights
    gino_model.eval()
    fm_model.eval()
    torch.save(gino_model.state_dict(), os.path.join(model_dir, f"gino_model_mcd.pt"))
    torch.save(fm_model.state_dict(), os.path.join(model_dir, f"fm_model_mcd.pt"))

print("===== Starting Evaluation =====")

gino_model.load_state_dict(torch.load(os.path.join(model_dir, "gino_model_mcd.pt"), weights_only=False), strict=True)
fm_model.load_state_dict(torch.load(os.path.join(model_dir, "fm_model_mcd.pt"), weights_only=False), strict=True)