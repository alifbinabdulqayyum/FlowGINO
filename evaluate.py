from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.functional.regression import continuous_ranked_probability_score as crps
import numpy as np
import torch
import os
from tqdm import tqdm
import random
from einops import rearrange

import argparse

# Create the parser
parser = argparse.ArgumentParser(
    description="Training Polymer Property Predictor"
)

parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--x-points", type=int, required=True)
parser.add_argument("--y-points", type=int, required=True)
parser.add_argument("--xtarget-points", type=int, required=True)
parser.add_argument("--ytarget-points", type=int, required=True)
parser.add_argument("--t-points", type=int, required=True)
parser.add_argument("--result-save-dir", type=str, required=True)
parser.add_argument("--eval-save-dir", type=str, required=True)

# Parse the arguments from the command line
args = parser.parse_args()

eval_dict = {}

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

lpips = LPIPS(net_type='alex', reduction='mean')

result_save_dir = os.path.join(
    args.result_save_dir, 
    f"seed-{args.seed}", 
    f"xy({args.x_points}-{args.y_points})->({args.xtarget_points}-{args.ytarget_points})",
    f"t-{args.t_points}",
)

pred_filepath = os.path.join(result_save_dir, 'FMGINO-PREDS-MCD-AUEU.npz')
data_fmgino = np.load(pred_filepath)

ssim_list, mse_list, lpips_list, crps_list = [], [], [], []

for target, prediction in tqdm(zip(data_fmgino['target'], data_fmgino['prediction'])):
    pred = prediction.mean(axis=(0,1))

    ssim_list.append(ssim(target, pred, data_range=2.0))
    mse_list.append(mean_squared_error(target, pred))
    lpips_list.append(
        lpips(
            torch.tensor(pred[None,None,:,:].repeat(3, axis=1)),
            torch.tensor(target[None,None,:,:].repeat(3, axis=1))
        ).item()
    )
    crps_list.append(
        crps(
            torch.tensor(rearrange(prediction, 'n m x y -> (x y) (n m)')), 
            torch.tensor(rearrange(target, 'x y -> (x y)'))).item()
    )

ssim_list = torch.tensor(ssim_list)
mse_list = 10*torch.log10(4.0/torch.tensor(mse_list))
lpips_list = torch.tensor(lpips_list)
crps_list = torch.tensor(crps_list)

eval_dict["FMGINO"] = {}

eval_dict["FMGINO"]["SSIM"] = ssim_list.numpy()
eval_dict["FMGINO"]["PSNR"] = mse_list.numpy()
eval_dict["FMGINO"]["LPIPS"] = lpips_list.numpy()
eval_dict["FMGINO"]["CRPS"] = crps_list.numpy()


pred_filepath = os.path.join(result_save_dir, 'FMVBLLGINO-PREDS-AUEU.npz')
data_fmpgino = np.load(pred_filepath)

ssim_list, mse_list, lpips_list, crps_list = [], [], [], []

for target, prediction in tqdm(zip(data_fmpgino['target'], data_fmpgino['prediction'])):
    pred = prediction.mean(axis=(0,1))

    ssim_list.append(ssim(target, pred, data_range=2.0))
    mse_list.append(mean_squared_error(target, pred))
    lpips_list.append(
        lpips(
            torch.tensor(pred[None,None,:,:].repeat(3, axis=1)),
            torch.tensor(target[None,None,:,:].repeat(3, axis=1))
        ).item()
    )
    crps_list.append(
        crps(
            torch.tensor(rearrange(prediction, 'n m x y -> (x y) (n m)')), 
            torch.tensor(rearrange(target, 'x y -> (x y)'))).item()
    )

ssim_list = torch.tensor(ssim_list)
mse_list = 10*torch.log10(4.0/torch.tensor(mse_list))
lpips_list = torch.tensor(lpips_list)
crps_list = torch.tensor(crps_list)

eval_dict["FMVBLLGINO"] = {}

eval_dict["FMVBLLGINO"]["SSIM"] = ssim_list.numpy()
eval_dict["FMVBLLGINO"]["PSNR"] = mse_list.numpy()
eval_dict["FMVBLLGINO"]["LPIPS"] = lpips_list.numpy()
eval_dict["FMVBLLGINO"]["CRPS"] = crps_list.numpy()

####

# pred_filepath = os.path.join(result_save_dir, 'GINO-PREDS.npz')
# data_gino = np.load(pred_filepath)

# ssim_list, mse_list, lpips_list = [], [], []

# for target, prediction in tqdm(zip(data_gino['target'], data_gino['prediction'])):
#     pred = prediction#.mean(axis=(0,1))

#     ssim_list.append(ssim(target, pred, data_range=2.0))
#     mse_list.append(mean_squared_error(target, pred))
#     lpips_list.append(
#         lpips(
#             torch.tensor(pred[None,None,:,:].repeat(3, axis=1)),
#             torch.tensor(target[None,None,:,:].repeat(3, axis=1))
#         ).item()
#     )

# ssim_list = torch.tensor(ssim_list)
# mse_list = 10*torch.log10(4.0/torch.tensor(mse_list))
# lpips_list = torch.tensor(lpips_list)

# eval_dict["GINO"] = {}

# eval_dict["GINO"]["SSIM"] = ssim_list.numpy()
# eval_dict["GINO"]["PSNR"] = mse_list.numpy()
# eval_dict["GINO"]["LPIPS"] = lpips_list.numpy()


# pred_filepath = os.path.join(result_save_dir, 'VBLLGINO-PREDS.npz')
# data_pgino = np.load(pred_filepath)

# ssim_list, mse_list, lpips_list = [], [], []

# for target, prediction in tqdm(zip(data_pgino['target'], data_pgino['prediction'])):
#     pred = prediction#.mean(axis=(0,1))

#     ssim_list.append(ssim(target, pred, data_range=2.0))
#     mse_list.append(mean_squared_error(target, pred))
#     lpips_list.append(
#         lpips(
#             torch.tensor(pred[None,None,:,:].repeat(3, axis=1)),
#             torch.tensor(target[None,None,:,:].repeat(3, axis=1))
#         ).item()
#     )

# ssim_list = torch.tensor(ssim_list)
# mse_list = 10*torch.log10(4.0/torch.tensor(mse_list))
# lpips_list = torch.tensor(lpips_list)

# eval_dict["VBLLGINO"] = {}

# eval_dict["VBLLGINO"]["SSIM"] = ssim_list.numpy()
# eval_dict["VBLLGINO"]["PSNR"] = mse_list.numpy()
# eval_dict["VBLLGINO"]["LPIPS"] = lpips_list.numpy()


# pred_filepath = os.path.join(result_save_dir, 'MMGN-PREDS.npz')
# data_mmgn = np.load(pred_filepath)

# ssim_list, mse_list, lpips_list = [], [], []

# for target, prediction in tqdm(zip(data_mmgn['target'], data_mmgn['prediction'])):
#     pred = prediction#.mean(axis=(0,1))

#     ssim_list.append(ssim(target, pred, data_range=2.0))
#     mse_list.append(mean_squared_error(target, pred))
#     lpips_list.append(
#         lpips(
#             torch.tensor(pred[None,None,:,:].repeat(3, axis=1)),
#             torch.tensor(target[None,None,:,:].repeat(3, axis=1))
#         ).item()
#     )

# ssim_list = torch.tensor(ssim_list)
# mse_list = 10*torch.log10(4.0/torch.tensor(mse_list))
# lpips_list = torch.tensor(lpips_list)

# eval_dict["MMGN"] = {}

# eval_dict["MMGN"]["SSIM"] = ssim_list.numpy()
# eval_dict["MMGN"]["PSNR"] = mse_list.numpy()
# eval_dict["MMGN"]["LPIPS"] = lpips_list.numpy()

####

filepath_dir = os.path.join(
    args.eval_save_dir, 
    f"seed-{args.seed}", 
    f"xy({args.x_points}-{args.y_points})->({args.xtarget_points}-{args.ytarget_points})",
    f"t-{args.t_points}",
    # f"evaluation.pkl"
)

os.makedirs(filepath_dir, exist_ok = True)

import pickle
with open(os.path.join(filepath_dir, f"evaluation.pkl"), 'wb') as file:
    pickle.dump(eval_dict, file)