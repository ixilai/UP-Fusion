from model.framework import UPFusion
import os
import torch.nn as nn
import torch
import cv2
import argparse
from tqdm import tqdm
import glob
import torch.nn.functional as F
from skimage import img_as_ubyte
import clip
#
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#
def hyper_args():
    parser = argparse.ArgumentParser(description='UP-Fusion framework train process')
    #
    parser.add_argument('--ir_path', default='./Test_image/Infrared', type=str)  # Infrared Image
    parser.add_argument('--vi_path', default='./Test_image/Visible', type=str)  # Visible Image
    parser.add_argument('--save_path', default='./Results', type=str)  # Results
    parser.add_argument("--Fusion_ckpt", default='./ckpt/Fuse.pth', help="path to pretrained model (deweather)")
    return parser.parse_args()
#
args = hyper_args()

clip_model, preprocess = clip.load("ViT-B/32", device='cuda')
clip_model.eval()
#
ir_path = args.ir_path
vi_path = args.vi_path
saving_path = args.save_path
Fusion_ckpt = args.Fusion_ckpt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Fusion_model = UPFusion().to(device)
Fusion_model = nn.DataParallel(Fusion_model)
checkpoint = torch.load(Fusion_ckpt, map_location=device)
Fusion_model.load_state_dict(checkpoint["AWF"],strict=False)
Fusion_model.eval()

with torch.no_grad():
    ir_path1 = glob.glob(ir_path + '/*')
    vi_path1 = glob.glob(vi_path + '/*')
    for path1, path2 in zip(tqdm(vi_path1), ir_path1):
        img_multiple_of = 8
        save_path = path1.replace(str(vi_path), str(saving_path))

        img_vi = cv2.imread(path1, cv2.IMREAD_COLOR)
        img_vi = cv2.cvtColor(img_vi, cv2.COLOR_BGR2RGB)
        if img_vi is not None and len(img_vi.shape) == 3:
            img_vi = torch.from_numpy(img_vi).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
        else:
            raise ValueError(f"Error processing img_vi from path: {path1}")

        img_ir = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        if img_ir is not None and len(img_ir.shape) == 2:
            img_ir = torch.from_numpy(img_ir).float().div(255.).unsqueeze(0).unsqueeze(0).to(device)
        else:
            raise ValueError(f"Error processing img_ir from path: {path2}")

        _, _, height, width = img_vi.shape

        H = (height + img_multiple_of - 1) // img_multiple_of * img_multiple_of
        W = (width + img_multiple_of - 1) // img_multiple_of * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0

        img_vi = F.pad(img_vi, (0, padw, 0, padh), 'reflect')
        img_ir = F.pad(img_ir, (0, padw, 0, padh), 'reflect')

        text_prompt = [
            'The image contains multiple views of the same scene captured from different modalities, each providing complementary information. Focus on preserving fine details,'
            ' semantic structures, and key regions that contribute to understanding the scene.']

        text_token = clip.tokenize(text_prompt).to(device)
        with torch.no_grad():
            text_code = clip_model.encode_text(text_token)

        text_code = text_code.to(torch.float32)
        Final = Fusion_model(img_vi, img_ir, text_code, epoch=None)

        restored = torch.clamp(Final, 0, 1)

        #
        restored = restored[:, :, :height, :width]
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = img_as_ubyte(restored[0])

        #
        save_path = save_path.replace(str(vi_path), str(saving_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
