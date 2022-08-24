from torch.utils.data import Dataset, DataLoader
import cv2
import os
import pandas as pd
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch
from torch import nn
import matplotlib.pyplot as plt
import glob

from model_env.configs import *




id2label = {0: '_background_', 1: 'back_bumper', 2: 'back_glass', 3: 'back_left_door', 4: 'back_left_light',
            5: 'back_right_door', 6: 'back_right_light', 7: 'front_bumper', 8: 'front_glass', 9: 'front_left_door',
            10: 'front_left_light', 11: 'front_right_door', 12: 'front_right_light', 13: 'hood', 14: 'left_mirror',
            15: 'right_mirror', 16: 'tailgate', 17: 'trunk', 18: 'wheel'}
label2id = {'_background_': 0, 'back_bumper': 1, 'back_glass': 2, 'back_left_door': 3, 'back_left_light': 4,
            'back_right_door': 5, 'back_right_light': 6, 'front_bumper': 7, 'front_glass': 8, 'front_left_door': 9,
            'front_left_light': 10, 'front_right_door': 11, 'front_right_light': 12, 'hood': 13, 'left_mirror': 14,
            'right_mirror': 15, 'tailgate': 16, 'trunk': 17, 'wheel': 18}
palette = np.array([[0, 0, 0],
                    [225, 76, 225],
                    [127, 255, 191],
                    [255, 193, 21],
                    [203, 203, 203],
                    [102, 178, 255],
                    [255, 196, 228],
                    [103, 202, 223],
                    [255, 71, 71],
                    [161, 255, 164],
                    [21, 253, 29],
                    [251, 255, 176],
                    [254, 148, 12],
                    [150, 0, 250],
                    [196, 0, 13],
                    [48, 166, 159],
                    [102, 51, 0],
                    [245, 255, 240],
                    [171, 199, 56]])
classes = ['_background_', 'back_bumper', 'back_glass', 'back_left_door', 'back_left_light', 'back_right_door',
           'back_right_light', 'front_bumper', 'front_glass', 'front_left_door', 'front_left_light', 'front_right_door',
           'front_right_light', 'hood', 'left_mirror', 'right_mirror', 'tailgate', 'trunk', 'wheel']

model_finetune = SegformerForSemanticSegmentation.from_pretrained("model_env/segformer-carpaets-b3-1e4",
                                                                      ignore_mismatched_sizes=True,
                                                                      num_labels=len(id2label), id2label=id2label,
                                                                      label2id=label2id, reshape_last_stage=True)
model_finetune.to(DEVICE)

feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)

class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, transforms=None, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train
        self.transforms = transforms

        sub_path = "train" if self.train else "test"
        self.img_dir = os.path.join(self.root_dir, "images", sub_path)
        self.ann_dir = os.path.join(self.root_dir, "mask", sub_path)

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segmentation_map = cv2.imread(os.path.join(self.ann_dir, self.annotations[idx]))
        segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2GRAY)

        #         image = Image.open()
        #         segmentation_map = Image.open()

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=segmentation_map)
            # randomly crop + pad both image and segmentation map to same size
            encoded_inputs = self.feature_extractor(augmented['image'], augmented['mask'], return_tensors="pt")
        else:
            encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


def convert_mask(mask: np.array) -> np.array:
    for label, color in enumerate(palette):
        if label == 0:
            continue
        tup_where = np.where(mask == label)
        for i in range(len(tup_where[0])):
            mask[tup_where[0][i], tup_where[1][i], tup_where[2][i]] = color[tup_where[2][i]]
    return mask


def visualize_model(image,filename: str,model_finetune):
    filename = filename.split('.')

    inputs = feature_extractor(images=image, return_tensors="pt").to(DEVICE)

    outputs = model_finetune(**inputs)
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(logits,
                                                 size=image.shape[:-1],  # (height, width)
                                                 mode='bilinear',
                                                 align_corners=False)
    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    # list_prediction = [classes[idx] for idx in np.unique(seg.cpu().numpy())]
    list_prediction = {idx: classes[idx] for idx in np.unique(seg.cpu().numpy())}
    # list_label = [classes[idx] for idx in label]
    print("Predction : ", list_prediction)
    for label, color in enumerate(palette):
        color_seg[seg.cpu() == label, :] = color
    # color_seg = color_seg[..., ::-1]
    img_sementice = np.array(image) * 0.5 + color_seg * 0.5
    img_sementice = img_sementice.astype(np.uint8)


    cv2.imwrite(os.path.join('static', 'uploads', filename[0],'original.'+filename[-1]), image)
    cv2.imwrite(os.path.join('static', 'uploads', filename[0],'sementic.'+filename[-1]), img_sementice)
    cv2.imwrite(os.path.join('static', 'uploads', filename[0],'mask.'+filename[-1]), color_seg)


if __name__ == '__main__':
    print('Start')

    visualize_model('../train1.jpg')
    # img = cv2.imread('../train1.jpg')