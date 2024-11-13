import torch
import math
from eval import *
import argparse
from log_utils import *
from GMOD import GMOD
from torch.utils.data import DataLoader
from dataset import GMOD_Dataset

def validate(model, val_loader, criterion, device, epoch=0, exp_dir='./', ssim_required = False, val_mode = False):
    """ Function to print and log evaluation metrics : PSNR, SSIM, SAM & Loss """
    
    model.eval() # Setting to evaluation mode, stops dropout and normalizes with trained settings
    
    val_loss = 0.0
    PSNR_tot = 0

    # SSIM requires a lot of time to compute, so put it separately under an if statement
    if ssim_required:
        SSIM_tot = 0

    SAM_tot = 0
    best_psnr = 0
    best_sam = math.inf
    if ssim_required:
        best_ssim = 0

    SAM_times = 0
    PSNR_times = 0
    if ssim_required:
        SSIM_times = 0

    times = 0
    with torch.no_grad():
        for lr, hr, sr in val_loader:

            times+=1
            lr, hr, sr = lr.to(device), hr.to(device), sr.to(device)

            # Just making sure no NaNs screw things up
            assert not torch.isnan(lr).any(), "NaN detected in tensor"
            assert not torch.isnan(hr).any(), "NaN detected in tensor"
            assert not torch.isnan(sr).any(), "NaN detected in tensor"

            outputs = model(lr.clamp(0,1), sr.clamp(0,1))
            loss = criterion(outputs.clamp(0,1), hr.clamp(0,1))
            val_loss += loss.item()

            
            psnr_ = PSNR(np.clip(outputs.cpu().numpy(),0,1), np.clip(hr.cpu().numpy(), 0, 1))
            if psnr_ > best_psnr:
                best_psnr = psnr_

            if ssim_required:
                ssim_ = SSIM(np.clip(outputs.cpu().numpy(),0,1), np.clip(hr.cpu().numpy(), 0, 1))
                if ssim_ > best_ssim:
                    best_ssim = ssim_

            sam_ = SAM((outputs.cpu().numpy()), (hr.cpu().numpy()))
            if sam_ < best_sam:
                best_sam = sam_
            

            # Count values only if they are valid numbers 
            SAM_tot += sam_ if not math.isnan(sam_) else 0
            SAM_times += 1 if not math.isnan(sam_) else 0

            PSNR_tot += psnr_ if not math.isnan(psnr_) else 0
            PSNR_times += 1 if not math.isnan(psnr_) else 0

            if ssim_required:
                SSIM_tot += ssim_ if not math.isnan(ssim_) else 0
                SSIM_times += 1 if not math.isnan(ssim_) else 0

    loader_size = len(val_loader)

    # Printing & Logging
    if not val_mode:
        print(f"Validation Loss: {val_loss / loader_size:.4f}")
        if ssim_required:
            print(f" PSNR: {PSNR_tot/PSNR_times} ; SAM: {SAM_tot/SAM_times} ; SSIM: {SSIM_tot/SSIM_times}")
            print(f"Best PSNR : {best_psnr}, bestSAM: {best_sam}, bestSSIM: {best_ssim}")
            metrics = {'epoch':epoch, 'psnr':PSNR_tot/PSNR_times, 'ssim':ssim_, 'sam':SAM_tot/SAM_times, 'loss': val_loss / loader_size}
            log_validation_metrics(metrics, experiment_dir=exp_dir)
        else: 
            print(f" PSNR: {PSNR_tot/PSNR_times} ; SAM: {SAM_tot/SAM_times}")
            print(f"Best PSNR : {best_psnr}, bestSAM: {best_sam}")
            metrics = {'epoch':epoch, 'psnr':PSNR_tot/PSNR_times, 'sam':SAM_tot/SAM_times, 'loss': val_loss / loader_size}
            log_validation_metrics(metrics, experiment_dir=exp_dir)
    if val_mode:
        print(f" PSNR: {PSNR_tot/PSNR_times} ; SAM: {SAM_tot/SAM_times} ; SSIM: {SSIM_tot/SSIM_times}")
        




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sr3_weights_path', type=str, default='D:/Tarun_P/SR3_HSI/SR3_HSI/experiments/CHIKUSEI_2X/checkpoint/I100000_E142_gen.pth')
    parser.add_argument('--gmod_weights_path', type=str, default='GMOD_experiments/GMOD_experiment_20241114_011302/checkpoints/model_epoch_3.pth')
    parser.add_argument('--dataset', type=str, default='WashingtonDC')
    parser.add_argument('--use_eca', action='store_true') # Whether or not to use ECA (Efficient Channel Attention)
    parser.add_argument('--use_nonlocal', action='store_true') # Whether or not to use nonlocal conv blocks
    parser.add_argument('--l_res', type=int, default=32)
    parser.add_argument('--h_res', type=int, default=64)

    args = parser.parse_args()

    # Establishing the parameters first 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = args.dataset
    gmod_weights_path = str(args.gmod_weights_path)

    l_res = args.l_res
    h_res = args.h_res
    upscale_factor = int(h_res//l_res)
    batch_size = 4 # just hardcoding it into 4

    # Define Number of bands based on dataset
    if dataset=='Chikusei':
        num_bands = 128
    elif dataset=='WashingtonDC':
        num_bands = 191
    elif dataset=='Pavia':
        num_bands = 102
    else:
        num_bands = 3    

    # Load validation Dataset : 
    dataroot = r"D:/Tarun_P/Datasets"
    dataroot = f"{dataroot}/{dataset}/LR_HR/SR_{l_res}_{h_res}_{upscale_factor}x" # Adjust dataroot depending on your dataset path 
    val_dataset = GMOD_Dataset(dataroot, l_res, h_res, split='val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # Defining and loading the model 
    model = GMOD(num_bands, upscale_factor, use_ECA=args.use_eca, use_nonlocal=args.use_nonlocal)
    loaded_model = torch.load(gmod_weights_path)
    state_dict = loaded_model["model_state_dict"]
    model.load_state_dict(state_dict)

    criterion = torch.nn.MSELoss()

    # validate(model, train_loader, criterion, torch.device('cuda'), ssim_required=True, val_mode= True)
    validate(model=model, val_loader=val_loader, criterion=criterion, device=device, val_mode=True, ssim_required=False)
    params = sum(p.numel() for p in model.parameters())
    print(params) # just  logging param number 
