import os
import time
import copy
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import cv2
import torchvision.transforms as transforms

def colorize_labels(label_tensor, num_classes=10):
    """
    Convert label tensor to RGB color image.
    
    Args:
        label_tensor (torch.Tensor): Tensor of shape (H, W) containing class indices.
        num_classes (int): Number of classes.
    
    Returns:
        np.ndarray: RGB image of shape (H, W, 3).
    """
    # Define a colormap for the classes
    colormap = np.array([
        [0, 64, 0],      # Class 255: Dark Green
        [128, 0, 0],     # Class 0: Maroon
        [0, 128, 0],     # Class 1: Green
        [128, 128, 0],   # Class 2: Olive
        [0, 0, 128],     # Class 3: Navy
        [128, 0, 128],   # Class 4: Purple
        [0, 128, 128],   # Class 5: Teal
        [128, 128, 128], # Class 6: Gray
        [64, 0, 0],      # Class 7: Dark Red
        [0, 0, 0],       # Class 8: Black
    ], dtype=np.uint8)
    
    # Ensure the tensor is on the CPU and convert to numpy
    label_np = label_tensor.cpu().numpy()

    # Map each class index to its corresponding color
    colorized_label = colormap[label_np]
    return colorized_label
    

def debug_export_before_forward(inputs, labels, idx):
    # im = inputs[0]*255;
    im = inputs[0]
    im = im.to('cpu').numpy()
    im[0, :, :] = im[0, :, :] * 0.2016 + 0.5757
    im[1, :, :] = im[1, :, :] * 0.1678 + 0.4710
    im[2, :, :] = im[2, :, :] * 0.1222 + 0.3271
    im = im * 255
    im = im.astype(np.uint8)
    la = labels[0].to(torch.uint8).to('cpu').numpy()
    colorized_label = colorize_labels(la)
    im = im.transpose([1, 2, 0])
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{idx:06}_la.png", cv2.cvtColor(colorized_label, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{idx:06}_im.png", im)


def iou(pred, target, n_classes = 3):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("255")
    for cls in range(n_classes):
        if cls == 255:
            continue

        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union > 0:
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious)


def train_model(model, num_classes, dataloaders, criterion, optimizer, device, dest_dir, num_epochs=25):
    since = time.time()
    val_acc_history = []
    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    counter = 0

    writer = SummaryWriter(log_dir=os.path.join(dest_dir, "logs"))

    output_dir = os.path.join(dest_dir, "outputs")
    labels_dir = os.path.join(output_dir, "labels")
    predictions_dir = os.path.join(output_dir, "predictions")
    originals_dir = os.path.join(output_dir, "originals")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(originals_dir, exist_ok=True)

    to_tensor = transforms.ToTensor()

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}\n" + "-" * 10)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_iou_means = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                if inputs.shape[0] == 1:
                    continue
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)["out"]
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_iou_means.append((preds == labels).float().mean().item())

                if phase == "val":
                    colorized_labels = colorize_labels(labels[0], num_classes)
                    colorized_preds = colorize_labels(preds[0], num_classes)
                    
                    cv2.imwrite(os.path.join(labels_dir, f"{counter:06}_label.png"), cv2.cvtColor(colorized_labels, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(predictions_dir, f"{counter:06}_pred.png"), cv2.cvtColor(colorized_preds, cv2.COLOR_RGB2BGR))
                    
                    im = inputs[0].cpu().numpy()
                    mean = np.array([0.5757, 0.4710, 0.3271]).reshape(3, 1, 1)
                    std = np.array([0.2016, 0.1678, 0.1222]).reshape(3, 1, 1)
                    im = im * std + mean
                    im = (im * 255).clip(0, 255).astype(np.uint8)
                    im = im.transpose(1, 2, 0)
                    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(os.path.join(originals_dir, f"{counter:06}_original.png"), im)

                    writer.add_image("Original", to_tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), global_step=counter)
                    writer.add_image("Ground Truth", to_tensor(colorized_labels), global_step=counter)
                    writer.add_image("Prediction", to_tensor(colorized_preds), global_step=counter)

                counter += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = np.mean(running_iou_means) if running_iou_means else 0.0

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"Accuracy/{phase}", epoch_acc, epoch)

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state_dict = copy.deepcopy(model.state_dict())

        print()
    
    writer.close()
    return best_model_state_dict, val_acc_history
    