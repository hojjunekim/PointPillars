import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os

from dataset import PLR
from dataset.dataloader_plr import get_dataloader_plr
import argparse
from tqdm import tqdm
import pdb

from utils import setup_seed
from model import PointPillars
from loss import Loss
from torch.utils.tensorboard import SummaryWriter

import wandb
wandb.init(project='point pillar')

def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)

def main(args):
    setup_seed()
    wandb.config.update(args)
    
    train_dataset = PLR(data_root=args.data_root,
                          split='train')
    val_dataset = PLR(data_root=args.data_root,
                        split='val')
    train_dataloader = get_dataloader_plr(dataset=train_dataset, 
                                      batch_size=args.batch_size, 
                                      num_workers=args.num_workers,
                                      shuffle=True)
    val_dataloader = get_dataloader_plr(dataset=val_dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers,
                                    shuffle=False)

    if not args.no_cuda:
        pointpillars = PointPillars(nclasses=args.nclasses).cuda()
    else:
        pointpillars = PointPillars(nclasses=args.nclasses)
    loss_func = Loss()

    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(params=pointpillars.parameters(), 
                                  lr=init_lr, 
                                  betas=(0.95, 0.99),
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  
                                                    max_lr=init_lr*10, 
                                                    total_steps=max_iters, 
                                                    pct_start=0.4, 
                                                    anneal_strategy='cos',
                                                    cycle_momentum=True, 
                                                    base_momentum=0.95*0.895, 
                                                    max_momentum=0.95,
                                                    div_factor=10)
    saved_logs_path = os.path.join(args.saved_path, 'summary')
    os.makedirs(saved_logs_path, exist_ok=True)
    writer = SummaryWriter(saved_logs_path)
    saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints')
    os.makedirs(saved_ckpt_path, exist_ok=True)

    for epoch in range(args.max_epoch):
        print('=' * 20, epoch, '=' * 20)
        train_step, val_step = 0, 0
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            if not args.no_cuda:
                # move the tensors to the cuda
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].cuda()
            
            optimizer.zero_grad()

            batched_pts = data_dict['batched_pts']
            batched_gt_2d_pts = data_dict['batched_gt_2d_pts']
            batched_gt_labels = data_dict['batched_gt_labels']
            output = \
                pointpillars(batched_pts=batched_pts, 
                             mode='train',
                             batched_gt_2d_pts=batched_gt_2d_pts, 
                             batched_gt_labels=batched_gt_labels)
            
            loss_fn = nn.MSELoss()
            
            batched_range = torch.zeros((args.batch_size, 360), device='cuda')
            for i in range(len(batched_gt_2d_pts)-1):
                for j in range(len(batched_gt_2d_pts[i])):
                    batched_range[i][j] = torch.norm(batched_range[i][j])
                

            loss = loss_fn(output, batched_range)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step = epoch * len(train_dataloader) + train_step + 1

            if global_step % args.log_freq == 0:
                wandb.log({'training loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
            #     save_summary(writer, loss, global_step, 'train',
            #                  lr=optimizer.param_groups[0]['lr'], 
            #                  momentum=optimizer.param_groups[0]['betas'][0])
            train_step += 1
        if (epoch + 1) % args.ckpt_freq_epoch == 0:
            torch.save(pointpillars.state_dict(), os.path.join(saved_ckpt_path, f'epoch_{epoch+1}.pth'))

        if epoch % 2 == 0:
            continue
        pointpillars.eval()
        with torch.no_grad():
            for i, data_dict in enumerate(tqdm(val_dataloader)):
                if not args.no_cuda:
                    # move the tensors to the cuda
                    for key in data_dict:
                        for j, item in enumerate(data_dict[key]):
                            if torch.is_tensor(item):
                                data_dict[key][j] = data_dict[key][j].cuda()
                
                batched_pts = data_dict['batched_pts']
                batched_gt_2d_pts = data_dict['batched_gt_2d_pts']
                batched_gt_labels = data_dict['batched_gt_labels']
                output = \
                    pointpillars(batched_pts=batched_pts, 
                                mode='train',
                                batched_gt_2d_pts=batched_gt_2d_pts, 
                                batched_gt_labels=batched_gt_labels)
                
                global_step = epoch * len(val_dataloader) + val_step + 1
                if global_step % args.log_freq == 0:
                    wandb.log({'validation loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
                #     save_summary(writer, loss_dict, global_step, 'val')
                val_step += 1
        pointpillars.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='/home/hojkim/plr/data/2024_04_28_23_26_09', 
                        help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=160)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--ckpt_freq_epoch', type=int, default=20)
    parser.add_argument('--no_cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    main(args)
