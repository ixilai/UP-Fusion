import os
import sys
import time
import datetime
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed
import kornia
from tqdm import tqdm
import clip

from model.framework import UPFusion
from utils22.loss_vif import fusion_loss_vif
from dataloader.fuse_data_vsm import GetDataset_type

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_hyperparams():
    parser = argparse.ArgumentParser(description='RobF Net training')
    # Dataset paths
    parser.add_argument('--ir_path', default='./ir', type=str)  # Put your infrared image
    parser.add_argument('--vi_path', default='./vi', type=str)  # Put your visible image

    # Training parameters
    parser.add_argument('--patch_size', default=192, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--nEpochs', default=200, type=int)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--cuda', action='store_false', help='Use cuda?')
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument('--optim_gamma', default=0.8)
    parser.add_argument('--optim_step', default=1)
    parser.add_argument('--weight_decay', default=0)
    parser.add_argument('--clip_grad_norm_value', default=1e-4)
    parser.add_argument('--interval', default=1)
    parser.add_argument('--ckpt_path', default=None)
    return parser.parse_args()


def init_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    return local_rank, device


def build_model(device, ckpt_path=None):
    model = UPFusion().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    if ckpt_path:
        state = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state['UPFusion'], strict=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.optim_step, gamma=args.optim_gamma)
    return model, optimizer, scheduler


def build_dataloader():
    dataset = GetDataset_type(
        split='train',
        size=args.patch_size,
        ir_path=args.ir_path,
        vi_path=args.vi_path,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True
    )
    return loader


def train():
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    train_loader = build_dataloader()
    total_batches = len(train_loader)

    torch.backends.cudnn.benchmark = True
    start_time = time.time()

    for epoch in range(args.start_epoch, args.nEpochs + 1):
        Fusion.train()
        Fusion.zero_grad()

        with tqdm(total=total_batches, desc=f"Epoch [{epoch}/{args.nEpochs}]", ncols=120) as pbar:
            for batch_idx, (data_IR, data_VIS) in enumerate(train_loader):
                data_VIS, data_IR = [
                    x.cuda(non_blocking=True) for x in (data_VIS, data_IR)
                ]

                text_prompt = [
                    'The image contains multiple views of the same scene captured from different modalities, '
                    'each providing complementary information. Focus on preserving fine details, semantic structures, '
                    'and key regions that contribute to understanding the scene.'
                ]
                text_tokens = clip.tokenize(text_prompt).to(device)
                with torch.no_grad():
                    text_code = clip_model.encode_text(text_tokens).to(torch.float32)

                output_fusion = Fusion(data_VIS, data_IR, text_code, epoch=epoch)

                loss_fn = fusion_loss_vif(device)
                loss_gradient, loss_l1_val = loss_fn(data_VIS, data_IR, output_fusion)
                loss = loss_l1_val + loss_gradient

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(Fusion.parameters(), max_norm=args.clip_grad_norm_value)
                optimizer.step()

                elapsed = time.time() - start_time
                batches_done = (epoch - 1) * total_batches + batch_idx + 1
                total_batches_all = args.nEpochs * total_batches
                progress_percent = batches_done / total_batches_all * 100
                eta_seconds = elapsed / batches_done * (total_batches_all - batches_done)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                pbar.set_postfix({
                    'loss': f"{loss.item():.6f}",
                    'Progress': f"{progress_percent:.2f}%",
                    'ETA': eta
                })
                pbar.update(1)

        if epoch % 10 == 0:
            os.makedirs('./ckpt', exist_ok=True)
            checkpoint = {'AWF': Fusion.state_dict()}
            torch.save(checkpoint, f'./ckpt/{epoch}.pth')

        scheduler.step()
        if optimizer.param_groups[0]['lr'] <= 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6



if __name__ == "__main__":
    args = get_hyperparams()
    local_rank, device = init_distributed()
    Fusion, optimizer, scheduler = build_model(device, ckpt_path=args.ckpt_path)
    train()
