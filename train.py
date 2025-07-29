import torch
import argparse
from config import (
    NUM_WORKERS, DEVICE, AMP,
    TRAIN_IMG, TRAIN_ANNOT, VALID_IMG, VALID_ANNOT,
    CLASSES, NUM_CLASSES, PROJECT_NAME
)
from model import create_model
from datasets import create_train_dataset, create_valid_dataset, create_train_loader, create_valid_loader
from utils import Averager, save_model, validate, get_train_transform, get_valid_transform
from tqdm.auto import tqdm

# Argument parser
parser = argparse.ArgumentParser(description='Train RetinaNet on custom dataset')
parser.add_argument('--epochs', type=int, default=75, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd', help='Optimizer type')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
parser.add_argument('--early_stopping_patience', type=int, default=None, help='Early stopping patience epochs (None disables early stopping)')
args = parser.parse_args()

print(f"[INFO] Training config -> Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}, Optimizer: {args.optimizer}, Weight decay: {args.weight_decay}, Early stopping patience: {args.early_stopping_patience}")

# Load datasets
train_dataset = create_train_dataset(TRAIN_IMG, TRAIN_ANNOT, get_train_transform())
valid_dataset = create_valid_dataset(VALID_IMG, VALID_ANNOT, get_valid_transform())

train_loader = create_train_loader(train_dataset, args.batch_size, NUM_WORKERS)
valid_loader = create_valid_loader(valid_dataset, args.batch_size, NUM_WORKERS)

print(f"[INFO] Number of training samples: {len(train_dataset)}")
print(f"[INFO] Number of validation samples: {len(valid_dataset)}")

model = create_model(num_classes=NUM_CLASSES)
model.to(DEVICE)

# Optimizer with weight decay
params = [p for p in model.parameters() if p.requires_grad]
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler() if AMP else None

# Training loop
train_loss_hist = Averager()
best_valid_map = 0.0
epochs_no_improve = 0

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

    # Step LR scheduler
    lr_scheduler.step()

    # Validation
    print("[INFO] Validating...")
    valid_map = validate(model, valid_loader, device=DEVICE)

    print(f"[RESULT] Epoch {epoch+1} Loss: {train_loss_hist.value:.4f}, Validation mAP: {valid_map:.4f}")

    # Save best model
    if valid_map > best_valid_map:
        print(f"[INFO] Best mAP improved from {best_valid_map:.4f} to {valid_map:.4f}. Saving model.")
        best_valid_map = valid_map
        epochs_no_improve = 0
        save_model(epoch, model, optimizer, f'{PROJECT_NAME}_best.pth')
    else:
        epochs_no_improve += 1

    # Save last model
    save_model(epoch, model, optimizer, f'{PROJECT_NAME}_last.pth')

    # Early stopping check
    if args.early_stopping_patience is not None and epochs_no_improve >= args.early_stopping_patience:
        print(f"[INFO] Early stopping triggered after {epoch+1} epochs with no improvement.")
        break

print(f"[INFO] Training complete. Best mAP: {best_valid_map:.4f}")
