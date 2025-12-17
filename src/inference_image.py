import torch
import argparse
import cv2
import os
import numpy as np
from utils import get_segment_labels, draw_segmentation_map, image_overlay
from PIL import Image
from config import ALL_CLASSES, LABEL_COLORS_LIST
from model import prepare_model

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input dir')
parser.add_argument(
    '--model',
    default='../outputs/model.pth',
    help='path to the model checkpoint'
)
args = parser.parse_args()

out_dir = os.path.join('..', 'outputs', 'inference_results')
os.makedirs(out_dir, exist_ok=True)

# Set computation device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = prepare_model(num_classes=len(ALL_CLASSES)).to(device)
ckpt = torch.load(args.model)
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

all_image_paths = os.listdir(args.input)
for i, image_path in enumerate(all_image_paths):
    print(f"Image {i+1}")
    # Read the image.
    image = Image.open(os.path.join(args.input, image_path))
    image = image.resize((512, 512))

    # Do forward pass and get the output dictionary.
    outputs = get_segment_labels(image, model, device)
    outputs = outputs['out']
    segmented_image = draw_segmentation_map(outputs)

    # Extract label map from outputs tensor for bounding box calculations
    label_map = torch.argmax(outputs.squeeze(), dim=0).cpu().numpy()

    final_image = image_overlay(image, segmented_image)

    # Iterate over each class present in the segmented image
    classes_present = np.unique(label_map)
    for class_index in classes_present:
        if class_index == 0:  # Skip background class
            continue
        mask = label_map == class_index
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(final_image, (x, y), (x+w, y+h), LABEL_COLORS_LIST[class_index], 2)
            cv2.putText(final_image, ALL_CLASSES[class_index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 128), 2)

    # Add image name annotation on the final image.
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final_image, f"Image Name: {image_path}", (10, 30), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Segmented image', final_image)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join(out_dir, image_path), final_image)

cv2.destroyAllWindows()
