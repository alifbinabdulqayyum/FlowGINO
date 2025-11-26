
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
import torch.nn as nn

EPSILON = 1e-4

# Necessary functions for interpolation of latent parameters

def linear_interpolate_parameters(p1: nn.Parameter, t1: float, p2: nn.Parameter, t2: float, t_new: float) -> torch.Tensor:
    """
    Performs linear interpolation between two PyTorch nn.Parameters.

    This function can be used to interpolate between the weights or parameters of
    a neural network at different points in time or different states.

    The formula for linear interpolation is:
    p(t_new) = p1 + ((t_new - t1) / (t2 - t1)) * (p2 - p1)

    Args:
        p1 (nn.Parameter): The starting parameter tensor at time t1.
        t1 (float): The starting timepoint.
        p2 (nn.Parameter): The ending parameter tensor at time t2.
        t2 (float): The ending timepoint.
        t_new (float): The new timepoint for which to interpolate the parameter.

    Returns:
        torch.Tensor: The interpolated parameter tensor at time t_new.
                      Note: This is a torch.Tensor, not an nn.Parameter,
                      as it is a result of a computation.
    """
    # Ensure the parameters have the same shape
    if p1.shape != p2.shape:
        raise ValueError("The two parameters must have the same shape.")

    # Calculate the interpolation factor
    # This factor determines how far between p1 and p2 the new parameter should be.
    interpolation_factor = (t_new - t1) / (t2 - t1)

    # Perform the linear interpolation using PyTorch tensor operations.
    # PyTorch handles the element-wise operations and broadcasting seamlessly.
    interpolated_param = p1 + interpolation_factor * (p2 - p1)

    return interpolated_param

def find_bounds(sorted_tensor_list, value:int):
    """
    Finds the lower and upper bounds for a given value within a sorted list.

    The function assumes the input list is sorted in ascending order.
    
    Args:
        sorted_tensor_list (list): A list of integers, sorted in ascending order.
        value (int): The integer for which to find the bounds.

    Returns:
        tuple: A tuple containing the lower and upper bounds. Returns (None, None)
               if the list is empty or the value is out of bounds.
    """
    
    # # Handle the case of an empty list
    # if not sorted_tensor_list:
    #     return (None, None)

    # If the value is smaller than the first element, the bounds are the first two elements.
    if value < sorted_tensor_list[0]:
        # Or you could return (None, sorted_tensor_list[0]) if you prefer.
        # This implementation returns the first two elements as a range.
        if len(sorted_tensor_list) > 1:
            return (sorted_tensor_list[0].item(), sorted_tensor_list[1].item())
        else:
            return (sorted_tensor_list[0].item(), None)
    
    # Iterate through the list to find the correct range
    for i in range(len(sorted_tensor_list) - 1):
        lower_bound = sorted_tensor_list[i]
        upper_bound = sorted_tensor_list[i + 1]
        
        if lower_bound <= value < upper_bound:
            return (lower_bound.item(), upper_bound.item())
            
    # Handle the case where the value is greater than or equal to the last element.
    # The upper bound is None since there is no element above it.
    if value >= sorted_tensor_list[-1]:
        return (sorted_tensor_list[-1].item(), None)

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
parser.add_argument("--result-save-dir", type=str, required=True)

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

# Get the train and test timestamps
tc_idx_train = torch.linspace(0, gst.shape[0]-1, t_points, dtype=int)
mask = torch.zeros_like(tc_idx, dtype=bool)
mask[tc_idx_train] = True
tc_idx_test = tc_idx[~mask]

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

# model_dir = "./saved_models_MMGN_full"

# result_save_dir = "./GST_results"
# os.makedirs(result_save_dir, exist_ok=True)

model_dir = os.path.join(
    args.model_dir, 
    f"seed-{seed}", 
    f"xy({x_points}-{y_points})->({xtarget_points}-{ytarget_points})",
    f"t-{t_points}",
    f"mmgn",
)

model_filepath = os.path.join(model_dir, "model.pt")
model.load_state_dict(torch.load(model_filepath, weights_only=False), strict=True)

print(f"Loaded Model Weights from: {model_filepath}")

old_latents = model.latents.detach()
new_latents = nn.Parameter(torch.zeros(size=(len(tc),old_latents.shape[-1])), requires_grad=False)

for old_idx, idx in enumerate(tc_idx_train):
    new_latents[idx] = old_latents[old_idx]

for t in tc_idx_test:
    t = t.item()
    lower_bound, upper_bound = find_bounds(tc_idx_train, t)
    # print(tc[lower_bound], tc[t], tc[upper_bound])
    new_latents[t] = linear_interpolate_parameters(
        new_latents[lower_bound], 
        tc[lower_bound], 
        new_latents[upper_bound], 
        tc[upper_bound], 
        tc[t]
    )

model.latents = new_latents

model.to(device=device)

result_save_dir = os.path.join(
    args.result_save_dir, 
    f"seed-{seed}", 
    f"xy({x_points}-{y_points})->({xtarget_points}-{ytarget_points})",
    f"t-{t_points}",
    # f"gino",
) 

print("===== Starting Evaluation =====")

predictions, inputs, targets = [], [], []
model.eval()
for batch in tqdm(val_loader):
    # Get batch data
    input_coords = batch["sparse_coords"]
    input_data = batch["sparse_observation"]

    target_coords = batch["target_coords"]
    target_data = batch["target_observation"]

    latent_idx = batch["idx"]

    # Forward pass through GINO
    with torch.no_grad():
        output = model(
            target_coords[...,1:3].to(device), 
            latent_idx.to(device)
        )
    predictions.append(output.reshape((xtarget_points, ytarget_points)).cpu().numpy())
    targets.append(target_data.reshape((xtarget_points, ytarget_points)).numpy())
    inputs.append(input_data.reshape((x_points, y_points)).numpy())

print("===== Saving Predictions =====")

np.savez(
    os.path.join(result_save_dir, 'MMGN-PREDS.npz'), 
    prediction=np.array(predictions), 
    target=np.array(targets),
    input=np.array(inputs),
)