import os
from datetime import datetime
import torch

def create_experiment_folder(experiment_name):
    base_dir = 'GMOD_experiments'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    os.makedirs(experiment_dir)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'))
    os.makedirs(os.path.join(experiment_dir, 'logs'))
    os.makedirs(os.path.join(experiment_dir, 'tb_log'))
    
    return experiment_dir

# Log Training
def log_model_params(model, experiment_dir):
    log_file = os.path.join(experiment_dir, 'logs', 'training.log')
    with open(log_file, 'w') as f:
        # Log number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        f.write(f"Total Parameters: {num_params}\n")
        f.write("Epoch\tLoss\n")

def log_training_loss(epoch, loss, experiment_dir):
    log_file = os.path.join(experiment_dir, 'logs', 'training.log')
    with open(log_file, 'a') as f:
        f.write(f"{epoch}\t{loss:.4f}\n")


def log_validation_metrics(metrics, experiment_dir, ssim_required = False):
    """ Function to log validation metrics to the experiment_dir defined via CMD. """
    log_file = os.path.join(experiment_dir, 'logs', 'val.log')
    with open(log_file, 'a') as f:
        # f.write("Epoch\t" + "\t".join(metrics.keys()) + "\n")
        # for epoch, vals in metrics.items():
        #     vals_str = "\t".join(f"{v:.4f}" for v in vals)
        #     f.write(f"{epoch}\t{vals_str}\n")
        if ssim_required: 
            validation_str = f"Epoch: {metrics["epoch"]}, PSNR: {metrics['psnr']}, SSIM: {metrics['ssim']}, SAM: {metrics['sam']}, RMSE: {metrics['loss']**(0.5)}\n"
        else: 
            validation_str = f"Epoch: {metrics["epoch"]}, PSNR: {metrics['psnr']}, SAM: {metrics['sam']}, RMSE: {metrics['loss']**(0.5)}\n"
        f.write(validation_str)


def save_checkpoint(model, optimizer, epoch, experiment_dir):
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)