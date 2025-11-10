import torch.utils.data
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import torchvision.transforms as transforms
import os
import glob
import numpy as np
from dataloader import transforms as T


def _imread(path):
    im_cv = Image.open(path).convert('L')
    # im_cv = cv2.imread(str(path), flags)
    im_cv = im_cv.resize((600,400), Image.ANTIALIAS)
    assert im_cv is not None, f"Image {str(path)} is invalid."
    # im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
    tran = transforms.ToTensor()
    im_ts = tran(im_cv) / 255.
    return im_ts

class GetDataset_type(torch.utils.data.Dataset):
    def __init__(self, split, size, ir_path=None, vi_path=None):
        super(GetDataset_type, self).__init__()

        if split == 'train':
            data_dir_ir = ir_path
            data_dir_vis = vi_path


            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)



            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

            self.transform = T.Compose([T.RandomCrop(size),
                                    T.RandomHorizontalFlip(0.5),
                                    T.RandomVerticalFlip(0.5),
                                    T.ToTensor()])

        elif split == 'val':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split=='train':
            # print('-----------')
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]

            image_vis = Image.open(vis_path).convert(mode='RGB')
            image_ir = Image.open(ir_path).convert(mode='L')
            image_vis, image_ir = self.transform(image_vis, image_ir)

            return (
                torch.tensor(image_ir),
                torch.tensor(image_vis),
                # torch.tensor(image_gt),
                # torch.tensor(image_gt_ir),
                # torch.tensor(image_gt_fuse)
            )

    def __len__(self):
        # print(self.length)
        return self.length

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames