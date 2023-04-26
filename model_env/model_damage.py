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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_model_damage():

    classes_damage = ('background', 'dent', 'scratch', 'crack', 'glass_shatter', 'lamp_broken', 'tire_flat',)
    palette = [[0, 0, 0], [129, 127, 38], [120, 69, 125], [53, 125, 34],
               [0, 11, 123], [118, 20, 12], [122, 81, 25]]
    id2label = dict([(idx, val) for idx, val in enumerate(classes_damage)])
    label2id = dict([(val, idx) for idx, val in enumerate(classes_damage)])

    model_damage = SegformerForSemanticSegmentation.from_pretrained("model_env/checkpoint-4550",
                                                                 ignore_mismatched_sizes=True,
                                                                 num_labels=len(id2label), id2label=id2label,
                                                                 label2id=label2id,
                                                                 reshape_last_stage=True)
    return model_damage


def convert_mask(mask: np.array) -> np.array:
    palette = [[0, 0, 0], [129, 127, 38], [120, 69, 125], [53, 125, 34],
               [0, 11, 123], [118, 20, 12], [122, 81, 25]]
    for label, color in enumerate(palette):
        if label == 0:
            continue
        tup_where = np.where(mask == label)
        for i in range(len(tup_where[0])):
            mask[tup_where[0][i], tup_where[1][i], tup_where[2][i]] = color[tup_where[2][i]]
    return mask

def visualize_model_damage(image, filename: str, model_finetune):
    filename = filename.split('.')
    classes_damage = ('background', 'dent', 'scratch', 'crack', 'glass_shatter', 'lamp_broken', 'tire_flat',)
    feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)


    inputs = feature_extractor(images=image, return_tensors="pt").to('cpu')

    outputs = model_finetune(**inputs)
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(logits,
                                                 size=image.shape[:-1],  # (height, width)
                                                 mode='bilinear',
                                                 align_corners=False)
    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    # list_prediction = [classes[idx] for idx in np.unique(seg.cpu().numpy())
    palette = [[0, 0, 0], [129, 127, 38], [120, 69, 125], [53, 125, 34],
               [0, 11, 123], [118, 20, 12], [122, 81, 25]]
    for label, color in enumerate(palette):
        color_seg[seg.cpu() == label, :] = color

    # color_seg = color_seg[..., ::-1]
    img_sementice = np.array(image) * 0.5 + color_seg * 0.5
    img_sementice = img_sementice.astype(np.uint8)


    cv2.imwrite(os.path.join('static', 'uploads_damage', filename[0], 'output.jpg'), np.array(seg))
    print(os.path.join('static', 'uploads_damage', filename[0], 'output.jpg'))
    cv2.imwrite(os.path.join('static', 'uploads_damage', filename[0], 'original.jpg'), image)
    cv2.imwrite(os.path.join('static', 'uploads_damage', filename[0], 'sementic.jpg'), img_sementice)
    cv2.imwrite(os.path.join('static', 'uploads_damage', filename[0], 'mask.jpg'), color_seg)



if __name__ == '__main__':
    classes_damage = ('background', 'dent', 'scratch', 'crack', 'glass_shatter', 'lamp_broken', 'tire_flat',)
    palette = [[0, 0, 0], [129, 127, 38], [120, 69, 125], [53, 125, 34],
               [0, 11, 123], [118, 20, 12], [122, 81, 25]]
    id2label = dict([(idx, val) for idx, val in enumerate(classes_damage)])
    label2id = dict([(val, idx) for idx, val in enumerate(classes_damage)])

    model_inf = SegformerForSemanticSegmentation.from_pretrained("checkpoint-4550",
                                                                 ignore_mismatched_sizes=True,
                                                                 num_labels=len(id2label), id2label=id2label,
                                                                 label2id=label2id,
                                                                 reshape_last_stage=True)

    feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
    img = cv2.imread('/Users/film/Documents/GitHub/car-part-segmentation-deployment-api/test_images/001168.jpg')
    visualize_model(img,'test.jpg',model_inf)

