import numpy as np
import cv2
import torch
import os
import time
import argparse
import pathlib

from model import create_model
from torchvision import transforms as transforms
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)
from utils.annotations import inference_annotations, annotate_fps

np.random.seed(42)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '--weights',
    default='outputs/best_model.pth',
    help='path to the model weights'
)
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='data/inference_data/video_1.mp4'
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

OUT_DIR = 'outputs/inference_outputs/videos'
os.makedirs(OUT_DIR, exist_ok=True)

# RGB format.
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the best model and trained weights.
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load(args.weights, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

cap = cv2.VideoCapture(args.input)

if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# Get the frame width and height.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = str(pathlib.Path(args.input)).split(os.path.sep)[-1].split('.')[0]
print(save_name)
# Define codec and create VideoWriter object .
out = cv2.VideoWriter(
    f"{OUT_DIR}/{save_name}.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), 30, 
    (frame_width, frame_height)
)

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)

# Read until end of video.
while(cap.isOpened()):
    # Capture each frame of the video.
    ret, frame = cap.read()
    if ret:
        image = frame.copy()
        height, width, _ = image.shape
        if args.imgsz is not None:
            image = cv2.resize(image, (args.imgsz, args.imgsz))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Apply transforms
        image_input = infer_transforms(image)
        image_input = torch.unsqueeze(image_input, 0)
        # Get the start time.
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
        print(f"Frame: {frame_count}")
        
        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # Carry further only if there are detected boxes.
        if len(outputs[0]['boxes']) != 0:
            frame = inference_annotations(
                outputs, 
                args.threshold, 
                CLASSES, 
                COLORS, 
                frame, 
                image,
                args
            )
        else:
            frame = frame
        frame = annotate_fps(frame, fps)

        out.write(frame)
        if args.show:
            cv2.imshow('image', frame)
            # Press `q` to exit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    else:
        break

# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()

# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")