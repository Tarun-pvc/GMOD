import argparse
import torch
from dataset import GMOD_Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from GMOD import GMOD
from losses import * 
from tqdm import tqdm
from test import validate
from log_utils import *
import math
from torch.utils.tensorboard import SummaryWriter
from adopt import ADOPT

gap = 10
from torch.cuda.amp import autocast, GradScaler
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model = model.to(device)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for lr, hr, sr in train_loader:
                lr, hr, sr = lr.to(device), hr.to(device), sr.to(device)
                assert not torch.isnan(lr).any(), "NaN detected in tensor"
                assert not torch.isnan(hr).any(), "NaN detected in tensor"
                assert not torch.isnan(sr).any(), "NaN detected in tensor"
                optimizer.zero_grad()

                with autocast():
                    outputs = model(lr, sr)
                    loss = criterion(outputs, hr)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.update() 
                running_loss += loss.item() if not math.isnan(loss.item()) else 0
                pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})
                pbar.update(1)
        log_training_loss(epoch, loss, experiment_dir=exp_dir)
        writer.add_scalar('Loss/train', loss.item(), epoch)

        if epoch % gap == 0:
            validate(model, val_loader, criterion, device, epoch, exp_dir)  
            # torch.save(model, f'./checkpoints/{dataset}_{scale}_Epoch{epoch}_10_Aug2024.pth')
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, experiment_dir= exp_dir)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device==torch.device('cuda'):
        print("cuda activated")

    # Dataset
    dataset = "WashingtonDC"
    l_resolution = 32
    h_resolution = 64
    scale = int(h_resolution//l_resolution)

    # Argparsing
    parser = argparse.ArgumentParser()

    # parser.add_argument('-c', '--config', type=str, default=f'config/sr_sr3_{l_resolution}_{h_resolution}_{dataset}.json',
    #                     help='JSON file for configuration')
    # parser.add_argument('-debug', '-d', action='store_true')

    parser.add_argument('--sr3_weights_path', type=str, default='D:\Tarun_P\SR3_HSI\SR3_HSI\experiments\CHIKUSEI_2X\checkpoint\I100000_E142_gen.pth') # Use if inference is to be done live, while training the model ( Needs editing in the Code, Currently incompatible )
    parser.add_argument('--dataset', type=str, default='Pavia') # WashingtonDC, Pavia, or Chikusei
    parser.add_argument('--epochs', type=int, default = 1) # Number of epochs for training
    parser.add_argument('--use_eca', action='store_true') # Whether or not to use ECA (Efficient Channel Attention)
    parser.add_argument('--use_nonlocal', action='store_true') # Whether or not to use nonlocal conv blocks
    parser.add_argument('--experiment_name', type=str, default='GMOD_experiment') # Whether or not to use nonlocal conv blocks
    parser.add_argument('--log_interval', type=int, default=10)
    args = parser.parse_args()

    exp_dir = create_experiment_folder(args.experiment_name)
    writer = SummaryWriter(log_dir=f'{exp_dir}/tb_log')

    dataset = args.dataset
    
    dataroot = r"D:/Tarun_P/Datasets"
    dataroot = f"{dataroot}/{dataset}/LR_HR/SR_{l_resolution}_{h_resolution}_{scale}x"
    sr3_weights_path = r"D:\Tarun_P\SR3_HSI\SR3_HSI\experiments\CHIKUSEI_2X\checkpoint\I100000_E142_gen.pth" 
    
    if dataset=='Chikusei':
        num_bands = 128
    elif dataset=='WashingtonDC':
        num_bands = 191
    elif dataset=='Pavia':
        num_bands = 102
    else:
        num_bands = 3
        
    print('num_bands: ', num_bands)

    upscale_factor = int(h_resolution//l_resolution)
    batch_size = 12
    num_epochs = args.epochs
    learning_rate = 1e-5

    # Dataset and DataLoader
    transform = ToTensor()
    train_dataset = GMOD_Dataset(dataroot, l_resolution, h_resolution)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=4)
    val_dataset = GMOD_Dataset(dataroot, l_resolution, h_resolution, split='val')
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    args = parser.parse_args()

    # Model, criterion, optimizer
    model = GMOD(num_bands, upscale_factor, use_ECA=args.use_eca, use_nonlocal=args.use_nonlocal)

    # writer.add_graph(model, (torch.randn(1, num_bands, l_resolution, l_resolution), torch.randn(1, 3, h_resolution, h_resolution)))

    criterion = nn.MSELoss()
    # criterion  = gradientLoss(batch_size, 8e-1, 1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = ADOPT(model.parameters(), lr=learning_rate)
    # optimizer = ADOPT(model.parameters(), lr=1e-3)


    # Training
    log_model_params(model, exp_dir)
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    writer.close()
    save_checkpoint(model=model, optimizer=optimizer, epoch=num_epochs, experiment_dir= exp_dir)
    validate(model=model, val_loader=val_loader, criterion=criterion, device=device, epoch=num_epochs, exp_dir=exp_dir, ssim_required=False)

    # torch.save(model, f'./checkpoints/{dataset}_{scale}_Epoch{args.epochs}_10_Aug2024.pth')
    # test(model)
