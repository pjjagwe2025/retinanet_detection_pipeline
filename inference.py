import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse

from model import create_model
from torchvision import transforms as transforms
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)
from utils.annotations import inference_annotations

np.random.seed(42)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '--weights',
    default='outputs/best_model.pth',
    help='path to the model weights'
)
parser.add_argument(
    '-i', '--input', 
    help='path to input image directory or a single image',
    required=True
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    help='image resize shape'
)
parser.add_argument(
    '--threshold',
    default=0.25,
    type=float,
    help='detection threshold'
)
parser.add_argument(
    '--show', 
    action='store_true',
    help='whether to visualize the results in real-time'
)
parser.add_argument(
    '-nlb', '--no-labels',
    dest='no_labels',
    action='store_true',
    help='do not show labels during on top of bounding boxes'
)
args = parser.parse_args()

OUT_DIR = 'outputs/inference_outputs/images'
os.makedirs(OUT_DIR, exist_ok=True)

# RGB format.
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the best model and trained weights.
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load(args.weights, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images

def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)

DIR_TEST = args.input
test_images = collect_all_images(DIR_TEST)
print(f"Test instances: {len(test_images)}")

for i in range(len(test_images)):
    # Get the image file name for saving output later on.
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    if args.imgsz is not None:
        image = cv2.resize(image, (args.imgsz, args.imgsz))
    print(image.shape)
    # BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply transforms
    image_input = infer_transforms(image)
    # Add batch dimension.
    image_input = torch.unsqueeze(image_input, 0)
    start_time = time.time()
    # Predictions
    with torch.no_grad():
        outputs = model(image_input.to(DEVICE))
    end_time = time.time()

    # Get the current fps.
    fps = 1 / (end_time - start_time)
    # Total FPS till current frame.
    total_fps += fps
    frame_count += 1

    # Load all detection to CPU for further operations.
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # Carry further only if there are detected boxes.
    if len(outputs[0]['boxes']) != 0:
        # Draw the bounding boxes and write the class name on top of it.
        orig_image = inference_annotations(
            outputs, 
            args.threshold, 
            CLASSES, 
            COLORS, 
            orig_image, 
            image,
            args
        )

    if args.show:
        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(1)
    cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
    print(f"Image {i+1} done...")
    print('-'*50)

print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()
# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")