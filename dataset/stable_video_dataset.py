import os
from glob import glob
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset

class StableVideoDataset(Dataset):
    def __init__(self, 
        video_data_dir, 
        max_num_videos=None,
        frame_hight=576, frame_width=1024, num_frames=14,
        is_reverse_video=True,
        random_seed=42,
        double_sampling_rate=False,
    ):  
        self.video_data_dir = video_data_dir
        video_names = sorted([video for video in os.listdir(video_data_dir) 
                    if os.path.isdir(os.path.join(video_data_dir, video))])
        
        self.length = min(len(video_names), max_num_videos) if max_num_videos is not None else len(video_names)
        
        self.video_names = video_names[:self.length]
        if double_sampling_rate:
            self.sample_frames = num_frames*2-1
            self.sample_stride = 2
        else:
            self.sample_frames = num_frames
            self.sample_stride = 1

        self.frame_width = frame_width
        self.frame_height = frame_hight
        self.pixel_transforms = transforms.Compose([
            transforms.Resize((self.frame_height, self.frame_width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.is_reverse_video=is_reverse_video
        np.random.seed(random_seed)
        
    def get_batch(self, idx):
        video_name = self.video_names[idx]
        video_frame_paths = sorted(glob(os.path.join(self.video_data_dir, video_name, '*.png')))
        start_idx = np.random.randint(len(video_frame_paths)-self.sample_frames+1)
        video_frame_paths = video_frame_paths[start_idx:start_idx+self.sample_frames:self.sample_stride]
        video_frames = [np.asarray(Image.open(frame_path).convert('RGB')).astype(np.float32)/255.0 for frame_path in video_frame_paths]
        video_frames = np.stack(video_frames, axis=0)
        pixel_values = torch.from_numpy(video_frames.transpose(0, 3, 1, 2))
        return pixel_values

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        conditions = pixel_values[-1]
        if self.is_reverse_video:
            pixel_values = torch.flip(pixel_values, (0,))
            
        sample = dict(pixel_values=pixel_values, conditions=conditions)
        return sample