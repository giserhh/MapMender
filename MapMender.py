import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

import warnings

warnings.simplefilter("ignore")

target_root_folder = r"test_data/location/target/"
results_root_folder = r"results/location/"

def print_message(message):
    print(message)

def calculate_mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def calculate_rmse(img1, img2):
    return np.sqrt(calculate_mse(img1, img2))

def calculate_nmse(img1, img2):
    mse = calculate_mse(img1, img2)
    var = np.var(img1.astype(np.float32))
    if var == 0:
        return float('inf')
    return mse / var

def calculate_mae(img1, img2):
    return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))

def calculate_metrics(target_path, result_path):
    target_img = cv2.imread(target_path)
    result_img = cv2.imread(result_path)
    if target_img is None or result_img is None:
        return None, "Cannot read map"
    
    if target_img.shape != result_img.shape:
        return None, "map dimensions do not match"
    
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    

    mse_val = calculate_mse(target_gray, result_gray)
    rmse_val = calculate_rmse(target_gray, result_gray)
    nmse_val = calculate_nmse(target_gray, result_gray)
    mae_val = calculate_mae(target_gray, result_gray)

    psnr_val = psnr(target_gray, result_gray, data_range=255)
    if np.isinf(psnr_val):
        psnr_val = 100

    ssim_val = ssim(target_gray, result_gray, data_range=255)

    return {
        'mse': mse_val,
        'rmse': rmse_val,
        'nmse': nmse_val,
        'mae': mae_val,
        'psnr': psnr_val,
        'ssim': ssim_val
    }, None

def process_result_folder(result_folder):
    result_folder_name = os.path.basename(result_folder)
    target_folder = os.path.join(target_root_folder, result_folder_name)
    if not os.path.exists(target_folder):
        return {
            'folder_name': result_folder_name,
            'mse_avg': 0,
            'rmse_avg': 0,
            'nmse_avg': 0,
            'mae_avg': 0,
            'psnr_avg': 0,
            'ssim_avg': 0,
            'error': f"Target folder {target_folder} does not exist"
        }
    
    target_files = [f for f in os.listdir(target_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    result_files = [f for f in os.listdir(result_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    map_names = [f for f in target_files if f in result_files]
    
    if not map_names:
        return {
            'folder_name': result_folder_name,
            'mse_avg': 0,
            'rmse_avg': 0,
            'nmse_avg': 0,
            'mae_avg': 0,
            'psnr_avg': 0,
            'ssim_avg': 0,
            'error': "No common map files found"
        }
    
    psnr_values = []
    ssim_values = []
    mse_values = []
    rmse_values = []
    nmse_values = []
    mae_values = []
    
    for map_name in map_names:
        target_path = os.path.join(target_folder, map_name)
        result_path = os.path.join(result_folder, map_name)
        
        metrics, error = calculate_metrics(target_path, result_path)
        
        if error:
            continue
        
        if metrics['psnr'] != 100:
            mse_values.append(metrics['mse'])
            rmse_values.append(metrics['rmse'])
            nmse_values.append(metrics['nmse'])
            mae_values.append(metrics['mae'])
            psnr_values.append(metrics['psnr'])
            ssim_values.append(metrics['ssim'])
    
    valid_psnr = [v for v in psnr_values if not np.isinf(v)]
    valid_mse = [v for v in mse_values if not np.isinf(v)]
    valid_rmse = [v for v in rmse_values if not np.isinf(v)]
    valid_nmse = [v for v in nmse_values if not np.isinf(v)]
    valid_mae = [v for v in mae_values if not np.isinf(v)]
    
    return {
        'folder_name': result_folder_name,
        'mse_avg': np.mean(valid_mse) if valid_mse else 0,
        'rmse_avg': np.mean(valid_rmse) if valid_rmse else 0,
        'nmse_avg': np.mean(valid_nmse) if valid_nmse else 0,
        'mae_avg': np.mean(valid_mae) if valid_mae else 0,
        'psnr_avg': np.mean(valid_psnr) if valid_psnr else 0,
        'ssim_avg': np.mean(ssim_values) if ssim_values else 0
    }

print("Calculating...")

result_folders = []
if any(os.path.isfile(os.path.join(results_root_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(results_root_folder)):
    result_folders.append(results_root_folder)
for item in os.listdir(results_root_folder):
    item_path = os.path.join(results_root_folder, item)
    if os.path.isdir(item_path) and any(os.path.isfile(os.path.join(item_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(item_path)):
        result_folders.append(item_path)

if not result_folders:
    print(f"Error: No folders containing maps found in {results_root_folder}")
    exit(1)

summary_results = []

for result_folder in result_folders:
    result = process_result_folder(result_folder)
    summary_results.append(result)

print_message(f"{'Name':<30}{'MSE':<15}{'RMSE':<15}{'NMSE':<15}{'MAE':<15}{'PSNR (dB)':<15}{'SSIM':<15}")
print_message("-" * 110)

for result in summary_results:
    if 'error' in result:
        print_message(f"{result['folder_name']:<30}{result['error']}")
    else:
        print_message(
            f"{result['folder_name']:<30}"
            f"{result['mse_avg']:<15.3f}{result['rmse_avg']:<15.3f}{result['nmse_avg']:<15.3f}"
            f"{result['mae_avg']:<15.3f}{result['psnr_avg']:<15.3f}{result['ssim_avg']:<15.3f}"
        )

print("\nCalculation completed.")
