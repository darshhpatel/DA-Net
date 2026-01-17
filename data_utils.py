import os
import random
import sys
import torch.utils.data as data
import torchvision.transforms as tfs
from PIL import Image
from torchvision.transforms import functional as FF
from metrics import *
from option import opt

sys.path.append('net')
sys.path.append('')
crop_size = 'whole_img'
if opt.crop:
    crop_size = opt.crop_size

class RS_Dataset(data.Dataset):
    #def __init__(self,path,train,size=crop_size,format='.png',hazy='cloud',GT='label'): # RICE
    #def __init__(self,path,train,size=crop_size,format='.png',hazy='input',GT='target'): # SATE 1K
    def __init__(self, path, train, size=crop_size, format='.png', hazy='hazy', GT='GT', clip_length=5):
        super(RS_Dataset, self).__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format
        self.clip_length = clip_length  # Number of frames per video clip
        hazy_dir_path = os.path.join(path, hazy)
        self.haze_imgs_dir = sorted([
            f for f in os.listdir(hazy_dir_path)
            if os.path.isfile(os.path.join(hazy_dir_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.haze_imgs = [os.path.join(hazy_dir_path, img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, GT)
        # Debug prints for test set loading
        print(f"[RS_Dataset] Folder: {hazy_dir_path}")
        print(f"[RS_Dataset] Found images: {len(self.haze_imgs)}")
        # Assume all frames in a folder are from a single video, sorted by name
        self.num_frames = len(self.haze_imgs)
        self.valid_indices = list(range(self.clip_length // 2, self.num_frames - self.clip_length // 2))
        print(f"[RS_Dataset] Valid indices: {len(self.valid_indices)} (clip_length={self.clip_length})")
    def __getitem__(self, index):
        # Adjust index to valid range for clips
        center = self.valid_indices[index]
        half = self.clip_length // 2
        indices = list(range(center - half, center + half + 1))
        haze_clip = []
        clear_clip = []
        for idx in indices:
            haze = Image.open(self.haze_imgs[idx])
            img = self.haze_imgs[idx]
            id = os.path.basename(img)
            # Map hazy_XXXX.png to clear_XXXX.png
            if id.startswith('hazy_'):
                clear_name = 'clear_' + id[len('hazy_'):]
            else:
                clear_name = id  # fallback if naming is already matching
            clear = Image.open(os.path.join(self.clear_dir, clear_name))
            clear = tfs.CenterCrop(haze.size[::-1])(clear)
            if not isinstance(self.size, str):
                i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
                haze = FF.crop(haze, i, j, h, w)
                clear = FF.crop(clear, i, j, h, w)
            haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
            haze_clip.append(haze)
            clear_clip.append(clear)
        # Stack to shape [T, C, H, W]
        haze_clip = torch.stack(haze_clip, dim=0)
        clear_clip = torch.stack(clear_clip, dim=0)
        return haze_clip, clear_clip
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.valid_indices)
