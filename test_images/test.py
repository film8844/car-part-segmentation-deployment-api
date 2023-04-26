from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch
from torch import nn
import matplotlib.pyplot as plt
import cv2
import numpy as np


def predict(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = feature_extractor(images=image, return_tensors="pt").to("cpu")
    outputs = model_inf(**inputs)
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(logits,
                        size=image.shape[:-1], # (height, width)
                        mode='bilinear',
                        align_corners=False)
    seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    overlay_pred = np.zeros_like(image)
    for c in palette_dict.keys():
        indices_pred = np.where(seg == c)
        overlay_pred[indices_pred] = palette_dict[c]
    blended_pred = cv2.addWeighted(image, 0.5, overlay_pred, 0.5, 0)
    plt.imsave('good.jpg',blended_pred)
    return blended_pred

if __name__ == '__main__':
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
    feature_extractor = SegformerFeatureExtractor(align=False, reduce_zero_label=False)

    image = cv2.imread("/Users/film/Documents/GitHub/car-part-segmentation-deployment-api/test_images/000783.jpg")
    predict(image)

