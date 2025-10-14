# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
# import pdb
import io
import time
import pandas as pd
import torchvision
import random
from PIL import Image, ImageOps
import cv2
import numbers
import math
import torch
from randaugment import RandAugment
import json
from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple, Union
from torchvision import transforms

import re
import matplotlib.pyplot as plt
import math

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                print(len(img_group))
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                print(len(img_group))
                rst = np.concatenate(img_group, axis=2)
                return rst

    
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


# class Action_DATASETS(data.Dataset):
#     def __init__(self, 
#                  list_file: str, 
#                  labels_file: str,
#                  num_segments: int = 1, 
#                  new_length: int = 1,
#                  image_tmpl: str ='img_{:05d}.jpg', 
#                  image_transform: Optional[Callable] = None,
#                  target_transform: Optional[Callable] = None,
#                  model_resolution: int = 224,
#                  random_shift: bool =True, 
#                  test_mode: bool=False, 
#                  index_bias: int=1, 
#                  height: int=224, 
#                  width: int=224, 
#                  expl_loss_weight: float = 0.15,
#                  expl_weight_according_to_mask_ratio: bool = True,
#                  label_box: bool=False, 
#                  debug: bool=False):

#         self.list_file = list_file
#         self.num_segments = num_segments
#         self.seg_length = new_length
#         self.image_tmpl = image_tmpl
#         self.random_shift = random_shift
#         self.test_mode = test_mode
#         self.loop=False
#         self.index_bias = index_bias
#         self.labels_file = labels_file
#         self.height = height
#         self.width = width
#         self.label_box = label_box
#         self.debug = debug
#         self.model_resolution = model_resolution
#         self.expl_loss_weight = expl_loss_weight
#         self.expl_weight_according_to_mask_ratio = expl_weight_according_to_mask_ratio

#         if self.index_bias is None:
#             if self.image_tmpl == "frame{:d}.jpg":
#                 self.index_bias = 0
#             else:
#                 self.index_bias = 1
#         self._parse_list()
#         self.initialized = False

#         self.image_transform = image_transform
#         self.target_transform = target_transform

#     def _load_image(self, directory, idx):
#         return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
    
#     def _load_teacher_box(self, annotation, idx):
#         box = []
#         if not self.label_box:
#             x1, y1, x2, y2 = torch.tensor(annotation['frames'][str(idx)]['teacher_box'])
#             x1 = int(torch.floor(x1))
#             y1 = int(torch.floor(y1))
#             x2 = int(torch.ceil(x2))
#             y2 = int(torch.ceil(y2))
#         else:
#             x1, y1, x2, y2 = torch.tensor(annotation['frames'][str(idx)]['label_box'])
#             x1 = int(torch.floor(x1))
#             y1 = int(torch.floor(y1))
#             x2 = int(torch.ceil(x2))
#             y2 = int(torch.ceil(y2))

#         return [x1, y1, x2, y2]


#     def set_mask_from_bb(self, mask_bb_indices, height, width, channels=3):
#         left, top, right, bottom = mask_bb_indices

#         mask_width = right - left
#         mask_height = bottom-top
        
#         mask = torch.zeros((1, channels, height, width))
#         mask[:, :, top:top+mask_height, left:left+mask_width] = 1.

#         return mask

#     def create_masks(self, mask_indices, height, width, channels=3):
#         if width < self.model_resolution:
#             toksX = 1
#         else:
#            toksX = width // self.model_resolution 
           
#         if height < self.model_resolution:
#             toksY = 1
#         else:
#             toksY = height // self.model_resolution
#         mask = self.set_mask_from_bb(mask_indices, height, width, channels=channels)

#         sideX, sideY = toksX * self.model_resolution, toksY * self.model_resolution
#         if sideX == 0:
#             sideX, sideY = self.model_resolution, self.model_resolution
#         # Resize masks to model resolution
#         mask = torch.nn.functional.interpolate(mask, (sideY, sideX))
#         mask = torch.round(mask)

#         return mask

#     def create_lambdas(self, mask):
#         lambda_val = self.expl_loss_weight
#         if self.expl_weight_according_to_mask_ratio:
#             mask = mask.to(dtype=torch.float)
#             lambda_val = lambda_val * (1 / mask.mean().sqrt())
        
#         return lambda_val
    
#     @property
#     def total_length(self):
#         return self.num_segments * self.seg_length
    
#     @property
#     def classes(self):
#         classes_all = pd.read_csv(self.labels_file)
#         return classes_all.values.tolist()
    
#     def _parse_list(self):
#         self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

#     def _sample_indices(self, record):
#         if record.num_frames <= self.total_length:
#             if self.loop:
#                 return np.mod(np.arange(
#                     self.total_length) + randint(record.num_frames // 2),
#                     record.num_frames) + self.index_bias
#             offsets = np.concatenate((
#                 np.arange(record.num_frames),
#                 randint(record.num_frames, size=self.total_length - record.num_frames)))
#             return np.sort(offsets) + self.index_bias
#         offsets = list()
#         ticks = [i * record.num_frames // self.num_segments
#                  for i in range(self.num_segments + 1)]

#         for i in range(self.num_segments):
#             tick_len = ticks[i + 1] - ticks[i]
#             tick = ticks[i]
#             if tick_len >= self.seg_length:
#                 tick += randint(tick_len - self.seg_length + 1)
#             offsets.extend([j for j in range(tick, tick + self.seg_length)])
#         return np.array(offsets) + self.index_bias

#     def _get_val_indices(self, record):
#         if self.num_segments == 1:
#             return np.array([record.num_frames //2], dtype=int) + self.index_bias
        
#         if record.num_frames <= self.total_length:
#             if self.loop:
#                 return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
#             return np.array([i * record.num_frames // self.total_length
#                              for i in range(self.total_length)], dtype=int) + self.index_bias
#         offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
#         return np.array([i * record.num_frames / self.num_segments + offset + j
#                          for i in range(self.num_segments)
#                          for j in range(self.seg_length)], dtype=int) + self.index_bias

#     def __getitem__(self, index):
#         record = self.video_list[index]
#         segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
#         return self.get(record, segment_indices)

#     def __call__(self, img_group):
#         return [self.worker(img) for img in img_group]

#     def adjust_bb(self, boxes, target_width, target_height):
#         # Compute the center of each bounding box
#         x_center = (boxes[:, 0] + boxes[:, 2]) / 2
#         y_center = (boxes[:, 1] + boxes[:, 3]) / 2

#         # Compute half-width and half-height, handling odd dimensions
#         half_width_left = np.floor(target_width / 2)
#         half_width_right = np.ceil(target_width / 2)
#         half_height_top = np.floor(target_height / 2)
#         half_height_bottom = np.ceil(target_height / 2)

#         # Create new bounding box coordinates
#         new_x0 = x_center - half_width_left
#         new_y0 = y_center - half_height_top
#         new_x1 = x_center + half_width_right
#         new_y1 = y_center + half_height_bottom

#         # Stack the new coordinates into the final tensor
#         new_boxes = torch.stack([new_x0, new_y0, new_x1, new_y1], dim=1)
#         return new_boxes, new_y1-new_y0, new_x1-new_x0


#     def get(self, record, indices):
#         images = list()
#         masks = list()
#         lambdas = list()
        
#         with open(os.path.join(record.path, 'annotation.json')) as f:
#             annotation = json.load(f, object_pairs_hook=OrderedDict)

#         for i, seg_ind in enumerate(indices):
#             p = int(seg_ind)
#             try:
#                 seg_imgs = self._load_image(record.path, p)
#                 width, height = seg_imgs[-1].size
#                 channels = 3
#                 bb = self._load_teacher_box(annotation, p)
#                 mask = self.create_masks(bb, height, width, channels=channels)
#                 lambda_val = self.create_lambdas(mask)
#             except OSError:
#                 print('ERROR: Could not read image "{}"'.format(record.path))
#                 print('invalid indices: {}'.format(indices))
#                 raise
#             images.extend(seg_imgs)
#             masks.append(mask)
#             lambdas.append(lambda_val)
            
#         if self.image_transform:
#             from torchvision.transforms import ToPILImage
#             # make the mask into an image so the image translations work
#             image_masks = []
#             pil_transform = ToPILImage()
#             for mask in masks:
#                 image_masks.append(pil_transform(mask.squeeze(dim=0)).convert('RGB'))
#             data = {'video': images, 'mask': image_masks}
#             data = self.image_transform(data)
#             process_data, image_masks = data['video'], data['mask']

#         return process_data, image_masks, lambdas, record.label
#         # return process_data, torch.stack(masks), lambdas, record.label

#     def __len__(self):
#         return len(self.video_list)



class EducationBBDataset(data.Dataset):
    def __init__(self, list_file, labels_file,
                 num_segments=1, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=1, 
                 height=224, width=224, label_box=False, debug=False,
                 bounding_boxes=True,caption_level=0):

        self.list_file = list_file
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop=False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.height = height
        self.width = width
        self.label_box = label_box
        self.debug = debug
        self.bounding_boxes = bounding_boxes
        self.caption_level = caption_level

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()
        self.initialized = False

    def _load_image(self, directory, idx):

        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
    
    def _load_teacher_box(self, annotation, idx):
        box = []
        if not self.label_box:
            box = torch.tensor(annotation['frames'][str(idx)]['teacher_box'])
        else:
            box = torch.tensor(annotation['frames'][str(idx)]['label_box'])

        return box
    
    def _load_caption(self, annotation, idx, level=0):
        return annotation['frames'][str(idx)]['caption'][str(level)]

    def create_masks(self, mask_indices, height, width, channels=3):
        if width < self.model_resolution:
            toksX = 1
        else:
           toksX = width // self.model_resolution 

        if height < self.model_resolution:
            toksY = 1
        else:
            toksY = height // self.model_resolution
        mask = self.set_mask_from_bb(mask_indices, height, width, channels=channels)

        sideX, sideY = toksX * self.model_resolution, toksY * self.model_resolution
        if sideX == 0:
            sideX, sideY = self.model_resolution, self.model_resolution
        # Resize masks to model resolution
        mask = torch.nn.functional.interpolate(mask, (sideY, sideX))
        mask = torch.round(mask)

        return mask
    
    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(
                    self.total_length) + randint(record.num_frames // 2),
                    record.num_frames) + self.index_bias
            offsets = np.concatenate((
                np.arange(record.num_frames),
                randint(record.num_frames, size=self.total_length - record.num_frames)))
            return np.sort(offsets) + self.index_bias
        offsets = list()
        ticks = [i * record.num_frames // self.num_segments
                 for i in range(self.num_segments + 1)]

        for i in range(self.num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record):
        if self.num_segments == 1:
            return np.array([record.num_frames //2], dtype=int) + self.index_bias
        
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
            return np.array([i * record.num_frames // self.total_length
                             for i in range(self.total_length)], dtype=int) + self.index_bias
        offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
        return np.array([i * record.num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=int) + self.index_bias

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        return self.get(record, segment_indices)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def adjust_bb(self, boxes, target_width, target_height):
        # Compute the center of each bounding box
        x_center = (boxes[:, 0] + boxes[:, 2]) / 2
        y_center = (boxes[:, 1] + boxes[:, 3]) / 2

        # Compute half-width and half-height, handling odd dimensions
        half_width_left = np.floor(target_width / 2)
        half_width_right = np.ceil(target_width / 2)
        half_height_top = np.floor(target_height / 2)
        half_height_bottom = np.ceil(target_height / 2)

        # Create new bounding box coordinates
        new_x0 = x_center - half_width_left
        new_y0 = y_center - half_height_top
        new_x1 = x_center + half_width_right
        new_y1 = y_center + half_height_bottom

        # Stack the new coordinates into the final tensor
        new_boxes = torch.stack([new_x0, new_y0, new_x1, new_y1], dim=1)
        return new_boxes, new_y1-new_y0, new_x1-new_x0

    def get(self, record, indices):
        images = list()
        bbs = list()
        captions = list()
        
        with open(os.path.join(record.path, 'annotation.json')) as f:
            bb_annotation = json.load(f, object_pairs_hook=OrderedDict)
        with open(os.path.join(record.path, 'new_annotation.json')) as f:
            txt_annotation = json.load(f, object_pairs_hook=OrderedDict)

        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)
            try:
                seg_imgs = self._load_image(record.path, p)
                bb = self._load_teacher_box(bb_annotation, p)
                # caption = self._load_caption(txt_annotation, p, level=self.caption_level)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            except:
                raise ValueError(record.path)
            images.extend(seg_imgs)
            bbs.append(bb)
            # captions.append(caption)

        cropped_images = []
        for i, (x0, y0, x1, y1) in enumerate(bbs):
            cropped_images.append(
                images[i].crop((
                    int(np.floor(x0)), 
                    int(np.ceil (y0)), 
                    int(np.floor(x1)), 
                    int(np.ceil (y1))
                ))
            )

        data = {'video': cropped_images, 'mask': None}
        data = self.transform(data)
        process_data, image_masks = data['video'], data['mask']
        
        data = {'video': images, 'mask': None}
        data = self.transform(data)
        aug_images, image_masks = data['video'], data['mask']

        nonaug_images = [transforms.ToTensor()(image) for image in images]
        nonaug_images = torch.stack(nonaug_images)
        
        bbs = torch.stack(bbs)

        # ff_caption = random.choice(captions)

        return process_data, aug_images, bbs, nonaug_images, None, record.label

    def __len__(self):
        return len(self.video_list)



class UCF10124Dataset(data.Dataset):
    def __init__(self,
                 root: str,
                 gt_pkl: str,
                 split: str = "train",
                 clip_len: int = 8,
                 transforms = None):
        """
        Args:
          root: root directory for “ucf101_24” data (contains rgb-images, brox-images).
          gt_pkl: path to UCF101v2-GT.pkl file.
          split: “train” or “test”.
          clip_len: number of frames to sample in a clip.
          transforms: transforms to apply on PIL images (or on stacked frames).
        """
        self.root = root
        self.clip_len = clip_len
        self.transforms = transforms

        # Load the GT pickle
        data = pd.read_pickle(gt_pkl) #'/standard/spencerNSF/VIVA/mmaction2/tools/data/ucf101_24/UCF101_v2/UCF101v2-GT.pkl')

        self.classes = [(i,c) for i, c in enumerate(data['labels'])]            # list of 24 class names
        self.labels = data['labels']
        self.gttubes = data['gttubes']          # dict: video → list of tubes
        self.nframes = data['nframes']          # dict: video → number of frames
        self.resolution = data['resolution']    # dict: video → (h, w)
        # train_videos/test_videos are lists-of-lists (nsplits)
        self.train_videos = data['train_videos']
        self.test_videos = data['test_videos']

        # Choose which split
        if split == "train":
            video_list = self.train_videos[0]  # first split
        else:
            video_list = self.test_videos[0]

        self.video_list = video_list

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        """
        Returns:
          frames: Tensor [T, C, H, W]
          target: dict with keys:
             - "boxes": list of length T, each is Tensor [n_objs, 4]
             - "labels": list of length T, each is Tensor [n_objs] (index into 0..23)
             - "video": video id (string)
        """
        vid = self.video_list[idx]  # e.g. "Basketball/v_Basketball_g01_c01"
        # number of frames in this video
        num_f = self.nframes[vid]
        # load all gttubes for this video (possibly multiple action instances)
        # Each tube is shape (n_frames, 5): [frame_idx, x1, y1, x2, y2]
        tubes = self.gttubes[vid][list(self.gttubes[vid].keys())[0]][0]

        # Sample a clip of length clip_len (you can do random or uniform)
        # For simplicity, pick first `clip_len` frames
        # Alternatively, pick a random contiguous segment
        start = 0
        end = min(len(tubes[:, 0].tolist()), self.clip_len)
        frame_indices = list(range(start, end))
        # raise ValueError(end, len(frame_indices), tubes.shape, len(tubes[:, 0].tolist()))
        # frame_indices = tubes[:, 0].tolist()

        frames = []
        # Prepare per-frame boxes & labels
        boxes_list = []
        labels_list = []
        for t in frame_indices:
            # Load image
            # Frame file names are zero-padded, e.g. “00001.jpg”
            img_path = os.path.join(self.root, "rgb-images", vid, f"{int(tubes[:, 0].tolist()[t]):05d}.jpg")
            img = Image.open(img_path).convert("RGB")
            frames.append(img)

            # Collect all boxes for this frame from all tubes
            all_boxes = []
            all_labels = []
            tube = tubes[t, :]
            # for tube in tubes:
            #     # tube is (n_frames, 5). We find if tube has an entry with frame_idx == t
            #     # Note: tube[:,0] are the frame indices
            #     # Option: assume tubes are ordered by frame
            #     # You might do a mask:
            # import pdb; pdb.set_trace()
            # mask = tube #(tube[:,0] == t)
            # if mask.any():
            # row = tube[mask][0]  # one row with [t, x1, y1, x2, y2]
            x1, y1, x2, y2 = tube[1], tube[2], tube[3], tube[4]
            all_boxes.append([x1, y1, x2, y2])
            # label is determined by which tube this is => tube index in tubes list
            label = self.labels.index(vid.split('/')[0])
            all_labels.append(label)
            if len(all_boxes) > 0:
                boxes_list.append(torch.tensor(all_boxes, dtype=torch.float32))
                labels_list.append(torch.tensor(all_labels, dtype=torch.int64))
            else:
                boxes_list.append(torch.zeros((0,4), dtype=torch.float32))
                labels_list.append(torch.zeros((0,), dtype=torch.int64))

        if len(frame_indices) < self.clip_len:
            for i in range(self.clip_len-len(frame_indices)):
                frames.append(frames[-1])
                boxes_list.append(boxes_list[-1])
                labels_list.append(labels_list[-1])

        # Stack frames: resulting shape [T, C, H, W]
        # frames = torch.stack(frames, dim=0)

        cropped_images = []
        # import pdb; pdb.set_trace()
        for i, box in enumerate(boxes_list):
            x0, y0, x1, y1 = box[0].tolist()
            cropped_images.append(
                frames[i].crop((
                    int(np.floor(int(x0))), 
                    int(np.ceil (int(y0))), 
                    int(np.floor(int(x1))), 
                    int(np.ceil (int(y1)))
                ))
            )

        process_data = []
        aug_images = []
        # for img, img_cropped in zip(frames, cropped_images):
        #     if self.transforms:
        data = {'video': cropped_images, 'mask': None}
        data = self.transforms(data)
        process_data, image_masks = data['video'], data['mask']
        
        data = {'video': frames, 'mask': None}
        data = self.transforms(data)
        aug_images, image_masks = data['video'], data['mask']

        nonaug_images = [transforms.ToTensor()(image) for image in frames]
        nonaug_images = torch.stack(nonaug_images)

        target = {
            "boxes": boxes_list,
            "labels": labels_list,
            "video": vid,
        }

        return process_data, aug_images, torch.stack(boxes_list).squeeze(dim=1), nonaug_images, vid, int(labels_list[0])