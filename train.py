import argparse
import torch
from utils.dataset import GMOD_Dataset
from torch.utils.data import DataLoader
from model.GMOD import GMOD
from model.losses import * 
from tqdm import tqdm
from test import validate
from utils.log_utils import *
from torch.utils.tensorboard import SummaryWriter
import math
from utils.adopt import ADOPT
from torch.amp import autocast, GradScaler
from torch.profiler import profile, ProfilerActivity


"""
1 - Incorrect normalization issues: Added batch normalization. Fixed the normalization function. 
2 - Inefficiency issues : ? 

Changes: 
1. Using Sigmoid instead of SoftMax for the same effect network.py line 59
2. Reducing attention usage
3. Align_corners = False GMOD.py line 64

"""


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model = model.to(device)
    scaler = GradScaler(device)

    # PROFILER FOR DEBUGGING (under construction)
    # with profile(
    #     activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     schedule = torch.profiler.schedule(wait=1,warmup=1,active=3, repeat=2), # Wait: number of epochs after which profiling starts. Warmup: Start tracing, discard results. When real tracing starts, no overhead. 
    #     on_trace_ready = torch.profiler.tensorboard_trace_handler('./profile_dir'),
    #     record_shapes = True, 
    #     profile_memory = True,
    #     with_stack = True
    # ) as prof:
        
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

                with autocast(device):
                    outputs = model(lr, sr)
                    loss = criterion(outputs, hr)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # trying to ensure cuda out of memory doesnt happen. 
                scaler.update() 
                running_loss += loss.item() if not math.isnan(loss.item()) else 0 
                pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})
                pbar.update(1)
                
                # prof.step()

        log_training_loss(epoch, loss, experiment_dir=exp_dir)
        writer.add_scalar('Loss/train', loss.item(), epoch)

        if epoch % log_interval == 0:
            validate(model, val_loader, criterion, device, epoch, exp_dir)  
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, experiment_dir= exp_dir)
            


if __name__ == "__main__":

    # Dataset
    l_resolution = 32
    h_resolution = 64
    scale = int(h_resolution//l_resolution)

    # Argparsing
    parser = argparse.ArgumentParser()

    parser.add_argument('--sr3_weights_path', type=str, default='D:/Tarun_P/SR3_HSI/SR3_HSI/experiments/CHIKUSEI_2X/checkpoint/I100000_E142_gen.pth') # Use if inference is to be done live, while training the model ( Needs editing in the Code, Currently incompatible )
    parser.add_argument('--dataset', type=str, default='WashingtonDC') # WashingtonDC, Pavia, or Chikusei
    parser.add_argument('--epochs', type=int, default = 1) # Number of epochs for training
    parser.add_argument('--use_eca', action='store_true') # Whether or not to use ECA (Efficient Channel Attention)
    parser.add_argument('--use_nonlocal', action='store_true') # Whether or not to use nonlocal conv blocks
    parser.add_argument('--experiment_name', type=str, default='GMOD_experiment') # Whether or not to use nonlocal conv blocks
    parser.add_argument('--batch_size', type=int, default=12) # Number of images per batch
    parser.add_argument('--log_interval', type=int, default=10) # Number of epochs between each log 
    parser.add_argument('--learning_rate', type=float, default=1e-5) # Set the learning rate
    parser.add_argument('--adopt', action='store_true') # use the ADOPT optimizer
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--use_attention', action='store_true')
    

    args = parser.parse_args()

    # Making a directory to store logs & outputs, setting tensorboard log directory
    exp_dir = create_experiment_folder(args.experiment_name)
    writer = SummaryWriter(log_dir=f'{exp_dir}/tb_log')

    ### Dataset setup. 

    # !!Change paths when needed!!
    dataset = args.dataset
    dataroot = r"D:/Tarun_P/Datasets"
    dataroot = f"{dataroot}/{dataset}/LR_HR/SR_{l_resolution}_{h_resolution}_{scale}x"
    sr3_weights_path = r"D:\Tarun_P\SR3_HSI\SR3_HSI\experiments\CHIKUSEI_2X\checkpoint\I100000_E142_gen.pth" 

    # Supports the following datasets.
    if dataset=='Chikusei':
        num_bands = 128
    elif dataset=='WashingtonDC':
        num_bands = 191
    elif dataset=='Pavia':
        num_bands = 102
    else:
        num_bands = 3
        
    print('Number of bands: ', num_bands) # For debugging and logging purposes

    # Dataset and DataLoader
    batch_size = args.batch_size
    train_dataset = GMOD_Dataset(dataroot, l_resolution, h_resolution)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=4) # prefetch_factor **
    val_dataset = GMOD_Dataset(dataroot, l_resolution, h_resolution, split='val')
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)


    ### Model Setup.
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    log_interval = args.log_interval

    model = GMOD(num_bands, l_resolution =  l_resolution, h_resolution=h_resolution, use_ECA=args.use_eca, use_nonlocal=args.use_nonlocal, use_attention = args.use_attention) # can I just pass the args? 

    # for tensorboard logging
    if args.tensorboard:
        writer.add_graph(model, (torch.randn(1, num_bands, l_resolution, l_resolution), torch.randn(1, 3, h_resolution, h_resolution)))
    
    # optimizer
    if args.adopt:
        optimizer = ADOPT(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # loss function 
    criterion = nn.MSELoss()


    ### Model Training.

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    if device==('cuda'):
        print("cuda activated")



    log_model_params(model, exp_dir)
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    writer.close()
    save_checkpoint(model=model, optimizer=optimizer, epoch=num_epochs, experiment_dir= exp_dir)
    validate(model=model, val_loader=val_loader, criterion=criterion, device=device, epoch=num_epochs, exp_dir=exp_dir, ssim_required=False)