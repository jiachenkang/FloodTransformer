import os
import time
import logging
import datetime
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import MSELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import wandb

from model import FloodTransformer
from dataset import FloodDataset


today = datetime.datetime.now().strftime('%Y%m%d%H')

CONFIG = {
    'exp_name': '48steps',
    'seed': 42,
    'batch_size': 4,
    'num_epochs': 400,
    'learning_rate': 1e-4,
    'lr_min': 5e-6,
    'weight_decay': 1e-3,
    'num_workers': 12,
    'pin_memory': True,
    'width': 768,
    'heads': 12,
    'layers': 12,
    'save_dir': f'checkpoints/{today}',
    'dtype': torch.bfloat16,
    'log_interval': 10,
    'loss_lambda_hw': 1.0,
    'loss_lambda_u': 0.5,
    'loss_lambda_v': 0.5,
    'available_gpus': '0',
    'start_time_interval': 12,
    'pred_length': 48,
    'end_step': 50,
    'neg_to_pos_ratio': 11/89,
    'water_level_scale': 1.0,
}


def setup_logging(rank, resume_wandb_id=None):
    if rank == 0:  # only record logs in main process
        os.makedirs(CONFIG['save_dir'], exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{CONFIG['save_dir']}/train.log"),
                logging.StreamHandler()
            ]
        )
        with open(f"{CONFIG['save_dir']}/config.txt", 'w') as f:
            for key, value in CONFIG.items():
                f.write(f"{key}: {value}\n")

        wandb_mode = 'online'
        wandb_kwargs = {
            "project": "CIN",
            "name": CONFIG['exp_name'],
            "config": CONFIG,
            "mode": wandb_mode
        }
        
        # If resuming, use the existing wandb ID
        if resume_wandb_id:
            wandb_kwargs["id"] = resume_wandb_id
            wandb_kwargs["resume"] = "must"
            logging.info(f"Resuming wandb run with ID: {resume_wandb_id}")
            
        wandb.init(**wandb_kwargs)

def setup(rank, world_size):
    """initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_TIMEOUT'] = '1200' 
    os.environ['NCCL_IB_TIMEOUT'] = '23' 

    dist.init_process_group(
        "nccl", 
        rank=rank, 
        world_size=world_size, 
        timeout=datetime.timedelta(seconds=1200)
        )

    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True

def setup_with_logging(rank, world_size, resume_wandb_id=None):
    """Setup both distributed environment and logging"""
    setup(rank, world_size)
    setup_logging(rank, resume_wandb_id)
    
def cleanup():
    """clean up distributed training environment"""
    dist.destroy_process_group()

def criterion(water_level_pred, has_water_pred, u_pred, v_pred, water_level_target, has_water_target, u_target, v_target):
    # mse_loss = MSELoss(reduction='none')  
    mse_loss = MSELoss()  
    bce_loss = BCEWithLogitsLoss(
        pos_weight=torch.tensor([CONFIG['neg_to_pos_ratio']], dtype=CONFIG['dtype']).to(water_level_pred.device)
    )

    wl_pred_sum = torch.sum(water_level_pred, dim=-1)
    wl_target_sum = torch.sum(water_level_target, dim=-1)

    water_level_loss = mse_loss(water_level_pred, water_level_target) + mse_loss(wl_pred_sum, wl_target_sum) / CONFIG['pred_length']
    has_water_loss = bce_loss(has_water_pred, has_water_target)
    u_loss = mse_loss(u_pred, u_target)
    v_loss = mse_loss(v_pred, v_target)

    return water_level_loss, has_water_loss, u_loss, v_loss

def save_checkpoint(model, optimizer, scheduler, epoch, loss, rank, is_best=False):
    """Save checkpoint"""
    if rank == 0:  # only save in main process
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }
        
        save_path = Path(CONFIG['save_dir'])
        save_path.mkdir(exist_ok=True)
        
        torch.save(checkpoint, save_path / 'latest.pt')
        if is_best:
            torch.save(checkpoint, save_path / 'best.pt')

def validate(model, val_loader, rank, dem_embed_gpu, side_lens_gpu, square_centers_gpu):
    """Validation function"""
    model.eval()
    total_water_level_loss = 0.0
    total_has_water_loss = 0.0
    total_u_loss = 0.0
    total_v_loss = 0.0

    dtype = CONFIG['dtype']
    
    # Add metrics tracking
    total_true_positives = 0.0
    total_false_positives = 0.0
    total_false_negatives = 0.0
    
    with torch.no_grad():
        for data, rain, u_target, v_target, water_level_target, has_water_target in val_loader:
            # move data to GPU and convert to correct dtype
            data = data.to(rank, dtype=dtype)
            rain = rain.to(rank, dtype=dtype)
            u_target = u_target.to(rank, dtype=dtype)
            v_target = v_target.to(rank, dtype=dtype)
            water_level_target = water_level_target.to(rank, dtype=dtype)
            has_water_target = has_water_target.to(rank, dtype=dtype)
            
            # forward pass
            water_level_pred, has_water_pred, u_pred, v_pred = model(
                data, 
                rain, 
                dem_embed_gpu, 
                side_lens_gpu, 
                square_centers_gpu
            )
            
            # compute loss
            water_level_loss, has_water_loss, u_loss, v_loss = criterion(water_level_pred, has_water_pred, u_pred, v_pred, water_level_target, has_water_target, u_target, v_target)
            total_water_level_loss += water_level_loss.item()
            total_has_water_loss += has_water_loss.item()
            total_u_loss += u_loss.item()
            total_v_loss += v_loss.item()
            
            # compute classification metrics with correct dtype
            pred_binary = (torch.sigmoid(has_water_pred) > 0.5).to(dtype=dtype)
            true_positives = (pred_binary * has_water_target).sum()
            false_positives = (pred_binary * (1 - has_water_target)).sum()
            false_negatives = ((1 - pred_binary) * has_water_target).sum()
            
            total_true_positives += true_positives.item()
            total_false_positives += false_positives.item()
            total_false_negatives += false_negatives.item()
    
    # compute average losses and metrics with correct dtype
    metrics = torch.tensor([
        total_water_level_loss,
        total_has_water_loss,
        total_u_loss,
        total_v_loss,
        total_true_positives,
        total_false_positives,
        total_false_negatives,
        len(val_loader)
    ], device=rank, dtype=dtype)
    
    
    dist.barrier()
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    total_water_level_loss, total_has_water_loss, total_u_loss, total_v_loss, \
    total_true_positives, total_false_positives, total_false_negatives, num_batches = metrics.tolist()
    
    avg_water_level_loss = total_water_level_loss / num_batches
    avg_has_water_loss = total_has_water_loss / num_batches
    avg_u_loss = total_u_loss / num_batches
    avg_v_loss = total_v_loss / num_batches
    precision = total_true_positives / (total_true_positives + total_false_positives + 1e-8)
    recall = total_true_positives / (total_true_positives + total_false_negatives + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    
    return avg_water_level_loss, avg_has_water_loss, avg_u_loss, avg_v_loss, precision, recall, f1_score



def train(rank, world_size, model, resume_path=None, resume_wandb_id=None):
    setup_with_logging(rank, world_size, resume_wandb_id)
    
    # convert model to specified dtype
    model = model.to(dtype=CONFIG['dtype'])
    
    # convert model to DDP model
    model = model.setup_ddp(rank)
    
    # create dataset
    trn_dataset = FloodDataset('data/trn', CONFIG)
    val_dataset = FloodDataset('data/val', CONFIG)
    
    # create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=CONFIG['learning_rate'], 
        weight_decay=CONFIG['weight_decay'], 
        betas=(0.9, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=CONFIG['num_epochs'], 
        eta_min=CONFIG['lr_min']
    )

    # move static data to GPU with specified dtype
    dem_embed_gpu = torch.load('data/dem_embeddings.pt', weights_only=True).to(rank, dtype=CONFIG['dtype'])
    side_lens_gpu = torch.load('data/side_lengths.pt', weights_only=True).to(rank)  
    square_centers_gpu = torch.load('data/square_centers.pt', weights_only=True).to(rank, dtype=CONFIG['dtype'])
    
    # create training and validation data loaders
    train_sampler = DistributedSampler(trn_dataset, shuffle=True, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        trn_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=train_sampler,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    
    val_sampler = DistributedSampler(val_dataset, shuffle=False, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=val_sampler,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )

    best_val_loss = float('inf')
    start_epoch = 0
    
    # Load checkpoint if resuming
    if resume_path is not None:
        if rank == 0:
            logging.info(f"Loading checkpoint from {resume_path}")
        
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(resume_path, map_location=map_location)
        
        # Load model weights
        model.module.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Set starting epoch and best validation loss
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        
        if rank == 0:
            logging.info(f"Resuming from epoch {start_epoch} with validation loss {best_val_loss}")
    
    # training loop
    for epoch in range(start_epoch, CONFIG['num_epochs']):
        # training stage
        model.train()
        train_sampler.set_epoch(epoch)
        
        epoch_water_level_loss = 0.0
        epoch_has_water_loss = 0.0
        start_time = time.time()
        
        if rank == 0:
            total_batches = len(train_loader)
            print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
            batch_start_time = time.time()

        for batch_idx, (data, rain, u_target, v_target, water_level_target, has_water_target) in enumerate(train_loader):
            # convert input data to specified dtype
            data = data.to(rank, dtype=CONFIG['dtype'])
            rain = rain.to(rank, dtype=CONFIG['dtype'])
            u_target = u_target.to(rank, dtype=CONFIG['dtype'])
            v_target = v_target.to(rank, dtype=CONFIG['dtype'])
            water_level_target = water_level_target.to(rank, dtype=CONFIG['dtype'])
            has_water_target = has_water_target.to(rank, dtype=CONFIG['dtype'])
            
            # forward pass
            water_level_pred, has_water_pred, u_pred, v_pred = model(
                data, 
                rain, 
                dem_embed_gpu, 
                side_lens_gpu, 
                square_centers_gpu
            )
            
            # compute loss
            water_level_loss, has_water_loss, u_loss, v_loss = criterion(water_level_pred, has_water_pred, u_pred, v_pred, water_level_target, has_water_target, u_target, v_target)
            loss = water_level_loss + has_water_loss * CONFIG['loss_lambda_hw'] + u_loss * CONFIG['loss_lambda_u'] + v_loss * CONFIG['loss_lambda_v']
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_water_level_loss += water_level_loss.item()
            epoch_has_water_loss += has_water_loss.item()
            
            # Progress tracking (only for rank 0)
            if rank == 0 and (batch_idx + 1) % CONFIG['log_interval'] == 0:
                batch_time = time.time() - batch_start_time
                avg_batch_time = batch_time / (batch_idx + 1)
                estimated_epoch_time = avg_batch_time * total_batches
                progress = (batch_idx + 1) / total_batches * 100
                
                print(f"Progress: {progress:.1f}% [{batch_idx + 1}/{total_batches}] "
                      f"Est. epoch time: {estimated_epoch_time/60:.1f}min "
                      f"left: {(estimated_epoch_time-batch_time)/60:.1f}min "
                      f"Loss: {loss.item():.4e} "
                      f"(WL: {water_level_loss.item():.4e}, HW: {has_water_loss.item():.4e})")
                
                # Batch-level wandb logging
                wandb.log({
                    "batch": epoch * len(train_loader) + batch_idx,
                    "batch_loss": loss.item(),
                    "batch_wl_loss": water_level_loss.item(),
                    "batch_hw_loss": has_water_loss.item(),
                    "batch_u_loss": u_loss.item(),
                    "batch_v_loss": v_loss.item()
                })
        
        # compute training average losses
        train_water_level_loss = epoch_water_level_loss / len(train_loader)
        train_has_water_loss = epoch_has_water_loss / len(train_loader)
        train_total_loss = train_water_level_loss + train_has_water_loss
        
        # validation stage
        val_wl_loss, val_hw_loss, val_u_loss, val_v_loss, val_precision, val_recall, val_f1 = validate(model, val_loader, rank, dem_embed_gpu, side_lens_gpu, square_centers_gpu)
        val_loss = val_wl_loss + val_hw_loss + val_u_loss + val_v_loss
        
        # update learning rate
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # save checkpoint (only if validation loss improves)
        is_best = val_wl_loss < best_val_loss
        if is_best:
            best_val_loss = val_wl_loss
            save_checkpoint(model, optimizer, scheduler, epoch, val_wl_loss, rank, is_best=True)
        
        # save latest checkpoint every epoch
        save_checkpoint(model, optimizer, scheduler, epoch, val_wl_loss, rank, is_best=False)
        
        # record epoch results
        if rank == 0:
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_total_loss,
                "train_wl_loss": train_water_level_loss,
                "train_hw_loss": train_has_water_loss,
                "val_loss": val_loss,
                "val_wl_loss": val_wl_loss,
                "val_hw_loss": val_hw_loss,
                "val_u_loss": val_u_loss,
                "val_v_loss": val_v_loss,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "learning_rate": current_lr,
                "epoch_time": time.time() - start_time
            }
            
            # log to wandb
            wandb.log(epoch_metrics)
            
            # log to log file
            logging.info(
                f'Epoch {epoch}/{CONFIG["num_epochs"]} | '
                f'Train Loss: {train_total_loss:.3e} '
                f'(WL: {train_water_level_loss:.3e}, HW: {train_has_water_loss:.3e}) | '
                f'Val Loss: {val_loss:.3e} | '
                f'(WL: {val_wl_loss:.3e}, HW: {val_hw_loss:.3e}, U: {val_u_loss:.3e}, V: {val_v_loss:.3e}) | '
                f'Val Pre: {val_precision:.3e} | '
                f'Val Rec: {val_recall:.3e} | '
                f'Val F1: {val_f1:.3e} | '
                f'LR: {current_lr:.3e} | '
                f'Time: {epoch_metrics["epoch_time"]:.2f}s'
            )
    
    if rank == 0:
        wandb.finish()
    cleanup()

def parse_args():
    parser = argparse.ArgumentParser(description='Train FloodTransformer model')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--wandb_id', type=str, default=None, help='wandb run ID to resume')
    parser.add_argument('--save_dir', type=str, default=None, help='directory to save checkpoints')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Update CONFIG based on command line arguments
    if args.save_dir:
        CONFIG['save_dir'] = args.save_dir
    elif args.resume:
        # If resuming and no save_dir specified, extract save_dir from resume path
        checkpoint_dir = Path(args.resume).parent
        if checkpoint_dir.name:  # Not an empty path
            CONFIG['save_dir'] = str(checkpoint_dir)

    # Set the GPUs you want to use
    os.environ['CUDA_VISIBLE_DEVICES'] = CONFIG['available_gpus']
    
    # set random seed
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    
    # get number of available GPUs
    world_size = torch.cuda.device_count()
    
    # create model
    model = FloodTransformer(
        context_length=47791,
        dem_input_dim=1280,
        rain_num_steps=48,
        width=CONFIG['width'],
        heads=CONFIG['heads'],
        layers=CONFIG['layers'],
        pred_length=CONFIG['pred_length']
    )
    
    # start multi-process training
    torch.multiprocessing.spawn(
        train,
        args=(world_size, model, args.resume, args.wandb_id),
        nprocs=world_size,
        join=True
    ) 