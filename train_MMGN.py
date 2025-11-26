
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
import numpy as np
import random
from einops import rearrange
from tqdm import tqdm
from models.MMGNet_net import MMGNet

EPSILON = 1e-4

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

# Hyperparameters
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
train_dataset = MMGN_Baseline(
    data=gst,
    coords=input_geom,
    input_idx_list=input_geom_idx_input,
    target_idx_list=input_geom_idx_target,
    input_points=int(x_points*y_points),
    target_points=int(xtarget_points*ytarget_points), #425,
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

device = torch.device("cuda")

model = MMGNet(
    in_size=2,
    n_data=t_points,
    hidden_size=256,
    latent_size=128,
    latent_init="zeros",
    out_size=1, 
    n_layers=5,
    input_scale=256,
    alpha=1,
    filter="Gabor"
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
    f"mmgn",
) 

os.makedirs(model_dir, exist_ok=True)

print("===== Starting Training =====")

pbar = tqdm(range(NUM_EPOCHS))

# from tqdm import tqdm

# # num_epochs = 400
# pbar = tqdm(range(NUM_EPOCHS))

for epoch in pbar:
    model.train()
    for batch in train_loader:
        # Get batch data
        input_coords = batch["sparse_coords"]
        input_data = batch["sparse_observation"]

        target_coords = batch["target_coords"]
        target_data = batch["target_observation"]

        latent_idx = batch["idx"]

        # Forward pass through GINO
        output = model(
            target_coords[...,1:3].to(device), 
            latent_idx.to(device)
        )

        # Calculate loss
        loss = loss_fn(output, target_data[...,None].to(device))
        # loss = output.train_loss_fn(target[:,:,None].to(device))

        
        # Update model weights
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step() 

        # Update learning rate
        # scheduler.step()
        # print(f"{loss.item():.4f}", end='\r')

        # Print loss
        pbar.set_description(f"Epoch: {epoch+1:04d}/{NUM_EPOCHS:04d} || Loss: {loss.item():08.8f}")
    
    # Save model weights
    model.eval()
    torch.save(model.state_dict(), os.path.join(model_dir, f"model.pt"))

print("===== Starting Evaluation =====")

model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), weights_only=False), strict=True)