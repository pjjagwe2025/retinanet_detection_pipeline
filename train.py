import os
import torch
import argparse
from config import (
    NUM_WORKERS, DEVICE, AMP,
    TRAIN_IMG, TRAIN_ANNOT, VALID_IMG, VALID_ANNOT,
    CLASSES, NUM_CLASSES, PROJECT_NAME, OUT_DIR
)
from model import create_model
from datasets import create_train_dataset, create_valid_dataset, create_train_loader, create_valid_loader
from utils import (
    save_model, validate, get_train_transform, get_valid_transform,
    save_loss_plot, save_mAP, coco_log, save_best_model
)
from tqdm.auto import tqdm

# --- Define Averager if missing from utils.py ---
class Averager:
    def __init__(self):
        self.reset()
    def reset(self):
        self.total = 0.0
        self.count = 0
    def update(self, val):
        self.total += val
        self.count += 1
    @property
    def value(self):
        return self.total / self.count if self.count > 0 else 0

# Argument parser
parser = argparse.ArgumentParser(description='Train RetinaNet on custom dataset')
parser.add_argument('--epochs', type=int, default=75, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd', help='Optimizer type')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
parser.add_argument('--early_stopping', type=int, default=0, help='Early stopping patience (0 = disable)')
args = parser.parse_args()

print(f"[INFO] Training config -> Epochs: {args.epochs}, Batch size: {args.batch_size}, "
      f"LR: {args.lr}, Optimizer: {args.optimizer}, Weight Decay: {args.weight_decay}, "
      f"Early Stopping Patience: {args.early_stopping}")

# Load datasets
train_dataset = create_train_dataset(TRAIN_IMG, TRAIN_ANNOT, get_train_transform())
valid_dataset = create_valid_dataset(VALID_IMG, VALID_ANNOT, get_valid_transform())

train_loader = create_train_loader(train_dataset, args.batch_size, NUM_WORKERS)
valid_loader = create_valid_loader(valid_dataset, args.batch_size, NUM_WORKERS)

print(f"[INFO] Number of training samples: {len(train_dataset)}")
print(f"[INFO] Number of validation samples: {len(valid_dataset)}")

# Create model
model = create_model(num_classes=NUM_CLASSES)
model.to(DEVICE)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# AMP support
scaler = torch.cuda.amp.GradScaler() if AMP else None

# Training loop
train_loss_list = []
map_list = []
map_50_list = []

train_loss_hist = Averager()
best_valid_map = 0.0
early_stop_counter = 0

for epoch in range(args.epochs):
    print(f"\n[INFO] Epoch {epoch+1}/{args.epochs}")
    train_loss_hist.reset()
    model.train()
    prog_bar = tqdm(train_loader, total=len(train_loader))

    for images, targets in prog_bar:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if AMP:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        train_loss_hist.update(losses.item())
        prog_bar.set_description(desc=f"Loss: {losses.item():.4f}")

    # Scheduler step
    scheduler.step()

    # Validation
    print("[INFO] Validating...")
    stats = validate(model, valid_loader, device=DEVICE)  # returns [map50, map]
    map_50, map_all = stats
    print(f"[RESULT] Epoch {epoch+1} Loss: {train_loss_hist.value:.4f}, mAP@0.5: {map_50:.4f}, mAP@0.5:0.95: {map_all:.4f}")

    train_loss_list.append(train_loss_hist.value)
    map_50_list.append(map_50)
    map_list.append(map_all)

    # Save best model
    save_best_model(model, float(stats[0]), epoch, OUT_DIR)

    # Save current model
    save_model(epoch, model, optimizer, OUT_DIR)

    # Save plots and logs
    save_loss_plot(OUT_DIR, train_loss_list)
    save_mAP(OUT_DIR, map_50_list, map_list)
    coco_log(OUT_DIR, stats)
    print('#' * 80)

    # Early stopping logic
    if map_all > best_valid_map:
        best_valid_map = map_all
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if args.early_stopping > 0 and early_stop_counter >= args.early_stopping:
            print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
            break

print(f"[INFO] Training complete. Best mAP: {best_valid_map:.4f}")
