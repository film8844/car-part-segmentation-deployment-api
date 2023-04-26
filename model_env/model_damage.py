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
    classes = ('background', 'dent', 'scratch', 'crack', 'glass_shatter', 'lamp_broken', 'tire_flat',)
    palette = [[0, 0, 0], [129, 127, 38], [120, 69, 125], [53, 125, 34],
               [0, 11, 123], [118, 20, 12], [122, 81, 25]]
    id2label = dict([(idx, val) for idx, val in enumerate(classes)])
    label2id = dict([(val, idx) for idx, val in enumerate(classes)])
    palette_dict = dict([(idx, val) for idx, val in enumerate(palette)])

    model_inf = SegformerForSemanticSegmentation.from_pretrained(
        "/Users/film/Documents/GitHub/car-part-segmentation-deployment-api/model_env/checkpoint-4550",
        ignore_mismatched_sizes=True,
        num_labels=len(id2label), id2label=id2label, label2id=label2id,
        reshape_last_stage=True).to("cpu")
    return model_inf


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
    classes = ('background', 'dent', 'scratch', 'crack', 'glass_shatter', 'lamp_broken', 'tire_flat',)
    palette = [[0, 0, 0], [129, 127, 38], [120, 69, 125], [53, 125, 34],
               [0, 11, 123], [118, 20, 12], [122, 81, 25]]
    palette_dict = dict([(idx, val) for idx, val in enumerate(palette)])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)
    inputs = feature_extractor(images=image, return_tensors="pt").to("cpu")
    outputs = model_finetune(**inputs)
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(logits,
                                                 size=image.shape[:-1],  # (height, width)
                                                 mode='bilinear',
                                                 align_corners=False)
    seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    overlay_pred = np.zeros_like(image)
    for c in palette_dict.keys():
        indices_pred = np.where(seg == c)
        overlay_pred[indices_pred] = palette_dict[c]
    blended_pred = cv2.addWeighted(image, 0.5, overlay_pred, 0.5, 0)
    print('New')
    plt.imsave(os.path.join('static', 'uploads_damage', filename[0], 'sementic.jpg'), blended_pred)
    plt.imsave(os.path.join('static', 'uploads_damage', filename[0], 'mask.jpg'), overlay_pred)



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

