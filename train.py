from config import (
    DEVICE, 
    NUM_CLASSES, 
    NUM_EPOCHS, 
    PROJECT_NAME,
    VISUALIZE_TRANSFORMED_IMAGES, 
    NUM_WORKERS,
    TRAIN_IMG,
    TRAIN_ANNOT,
    VALID_IMG,
    VALID_ANNOT,
    CLASSES,
    RESIZE_TO,
    BATCH_SIZE,
    LR,
    AMP,
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
    OUT_DIR = set_training_dir(PROJECT_NAME)
    SCALER = torch.cuda.amp.GradScaler() if AMP else None
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

    # Initialize the model and move to the computation device.
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
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, nesterov=True)
    scheduler = StepLR(
        optimizer=optimizer, step_size=50, gamma=0.1, verbose=True
    )

    # To store training loss and mAP values.
    train_loss_list = []
    map_50_list = []
    map_list = []

    # Whether to show transformed images from data loader or not.
    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils.general import show_tranformed_image
        show_tranformed_image(train_loader)

    # To save best model.
    save_best_model = SaveBestModel()

    # Training loop.
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        # Start timer and carry out training and validation.
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

        # save the best model till now.
        save_best_model(
            model, float(stats[0]), epoch, OUT_DIR
        )
        # Save the current epoch model.
        save_model(epoch, model, optimizer, OUT_DIR)

        # Save loss plot.
        save_loss_plot(OUT_DIR, train_loss_list)

        # Save mAP plot.
        save_mAP(OUT_DIR, map_50_list, map_list)
        scheduler.step()
        coco_log(OUT_DIR, stats)
        print('#'*80)