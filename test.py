import torch
import math
from eval import *
import argparse

def validate(model, val_loader, criterion, device, epoch, ssim_required = False):
    model.eval()
    
    val_loss = 0.0
    PSNR_tot = 0
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
            # if times >= 500:
            #     break
            times+=1
            lr, hr, sr = lr.to(device), hr.to(device), sr.to(device)
            assert not torch.isnan(lr).any(), "NaN detected in tensor"
            assert not torch.isnan(hr).any(), "NaN detected in tensor"
            assert not torch.isnan(sr).any(), "NaN detected in tensor"
            outputs = model(lr.clamp(0,1), sr.clamp(0,1))
            loss = criterion(outputs.clamp(0,1), hr.clamp(0,1))
            val_loss += loss.item()

            # if times == 0:
            #     print("shape of outputs and hr: ", outputs.shape, hr.shape)
                # times +=1 
            
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

            SAM_tot += sam_ if not math.isnan(sam_) else 0
            SAM_times += 1 if not math.isnan(sam_) else 0

            PSNR_tot += psnr_ if not math.isnan(psnr_) else 0
            PSNR_times += 1 if not math.isnan(psnr_) else 0

            if ssim_required:
                SSIM_tot += ssim_ if not math.isnan(ssim_) else 0
                SSIM_times += 1 if not math.isnan(ssim_) else 0

