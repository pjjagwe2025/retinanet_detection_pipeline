import argparse
from config import (
    DEVICE, 
    NUM_CLASSES, 
    PROJECT_NAME,
    VISUALIZE_TRANSFORMED_IMAGES, 
    NUM_WORKERS,
    TRAIN_IMG,
    TRAIN_ANNOT,
    VALID_IMG,
    VALID_ANNOT,
    CLASSES,
    RESIZE_TO,
    RESOLUTIONS
)
from model import create_model
from utils.general import (
    SaveBestModel, 
    save_model, 
    save_loss_plot,
    save_mAP,
    set_training_dir
)
from datasets import (
    create_train_dataset, 
    create_valid_dataset, 
    create_train_loader, 
    create_valid_loader
)
from torch.optim.lr_scheduler import StepLR
from engine import train, validate
from utils.logging import coco_log, set_log

import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import random

plt.style.use('ggplot')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='sgd')
    args = parser.parse_args()

    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience

    OUT_DIR = set_training_dir(PROJECT_NAME)
    SCALER = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    set_log(OUT_DIR)

    train_dataset = create_train_dataset(
        TRAIN_IMG, TRAIN_ANNOT, CLASSES, RESIZE_TO,
    )
    valid_dataset = create_valid_dataset(
        VALID_IMG, VALID_ANNOT, CLASSES, RESIZE_TO
    )
    train_loader = create_train_loader(train_dataset, BATCH_SIZE, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    if RESOLUTIONS is not None:
        min_size = tuple(RESOLUTIONS[i][0] for i in range(len(RESOLUTIONS)))
        max_size = RESOLUTIONS[-1][0]
    else:
        min_size, max_size = (RESIZE_TO, ), RESIZE_TO
    print(f"[INFO] Input image sizes to be randomly chosen: {RESOLUTIONS}")

    model = create_model(
        num_classes=NUM_CLASSES, 
        min_size=min_size, 
        max_size=max_size, 
    )
    model = model.to(DEVICE)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY)

    scheduler = StepLR(
        optimizer=optimizer, step_size=50, gamma=0.1, verbose=True
    )

    train_loss_list = []
    map_50_list = []
    map_list = []

    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils.general import show_tranformed_image
        show_tranformed_image(train_loader)

    save_best_model = SaveBestModel()
    best_map = 0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        start = time.time()
        train_loss = train(
            model, 
            train_loader, 
            optimizer, 
            DEVICE,
            scaler=SCALER,
        )
        stats = validate(model, valid_loader, DEVICE)
        print(f"Epoch #{epoch+1} train loss: {train_loss:.3f}")   
        print(f"Epoch #{epoch+1} mAP@0.50:0.95: {stats[0]}")
        print(f"Epoch #{epoch+1} mAP@0.50: {stats[1]}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        train_loss_list.append(train_loss)
        map_50_list.append(stats[1])
        map_list.append(stats[0])

        save_best_model(
            model, float(stats[0]), epoch, OUT_DIR
        )
        save_model(epoch, model, optimizer, OUT_DIR)
        save_loss_plot(OUT_DIR, train_loss_list)
        save_mAP(OUT_DIR, map_50_list, map_list)
        scheduler.step()
        coco_log(OUT_DIR, stats)

        if stats[0] > best_map:
            best_map = stats[0]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in mAP@0.50:0.95")
            break

        print('#'*80)
