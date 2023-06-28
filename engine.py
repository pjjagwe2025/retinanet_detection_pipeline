import torch
import time
import random
# import cv2
# import numpy as np

from metrics.coco_eval import CocoEvaluator
from metrics.coco_utils import get_coco_api_from_dataset
from metrics.iou_types import _get_iou_types
from metrics import utils
from tqdm.auto import tqdm
from torchvision.transforms.functional import resize

def adjust_image_size(image, target, new_size):
    # Calculate scaling factors
    old_width, old_height = image.shape[1], image.shape[2]
    new_width, new_height = new_size
    width_scale = new_width / old_width
    height_scale = new_height / old_height

    # Resize image
    image = resize(image, new_size)
    # np_image = np.ascontiguousarray(image.numpy().transpose([1, 2, 0]))

    # Adjust bounding boxes
    for box in target["boxes"]:
        x_min, y_min, x_max, y_max = box.tolist()
        x_min *= width_scale
        x_max *= width_scale
        y_min *= height_scale
        y_max *= height_scale
        # cv2.rectangle(np_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255), thickness=2)
        # cv2.imshow('Image', np_image)
        # cv2.waitKey(0)
        box = torch.tensor([x_min, y_min, x_max, y_max])

    return image, target

# Function for running training iterations.
def train(
    model, 
    train_dataloader, 
    optimizer, 
    device, 
    scaler=None, 
    resolutions=None
):
    print('Training')
    model.train()
    
    # Progress bar.
    prog_bar = tqdm(
        train_dataloader, 
        total=len(train_dataloader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        if resolutions is not None:
            # Choose a random resolution.
            new_res = random.choice(resolutions)
            # Adjust the image size and targets.
            images, targets = zip(*[adjust_image_size(image, target, new_res) for image, target in zip(images, targets)])
        
        size = f"{images[0].shape[1]}x{images[0].shape[2]}"
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Reduce losses over all GPUs for logging purposes.
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}, Size: {size}")
    return loss_value

# Function for running validation iterations.
def validate(model, valid_dataloader, device):
    print('Validating')
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(valid_dataloader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    model.eval()
    # Progress bar.
    prog_bar = tqdm(
        valid_dataloader, 
        total=len(valid_dataloader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        model_time = time.time()
        with torch.no_grad():
            outputs = model(images, targets)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return stats