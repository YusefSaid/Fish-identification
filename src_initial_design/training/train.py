import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import constants
from torch.cuda.amp import GradScaler, autocast


def train_snn(model: nn.Module,
              train_loader: DataLoader,
              optimizer: optim.Optimizer,
              num_epochs: int,
              device: torch.device,
              scaler: torch.cuda.amp.GradScaler):
    """
    Trains a Siamese Neural Network using contrastive loss.
    """
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_idx, ((images1, images2), labels) in enumerate(train_loader):
            images1, images2, labels = images1.to(device), images2.to(device), labels.float().to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                similarity = model(images1, images2)
                loss = nn.BCEWithLogitsLoss()(similarity, labels)

            if loss.item() == 0:
                #print("Warning: Loss is zero. Skipping step.")
                continue

            # prevent large update of gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Debugging prints for each batch
            print(
                f'\rEpoch:{epoch + 1}/{num_epochs}. Batch: {batch_idx}/{len(train_loader)} (batchSize:{constants.BATCH_SIZE}, Loss: {loss.item():.4f})', end='. ')

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")



def train_mask_cnn(model: nn.Module,
                   train_loader: DataLoader,
                   optimizer: optim.Optimizer,
                   num_epochs: int,
                   device: torch.device,
                   scaler):
    """
    Trains a Mask R-CNN model using the provided DataLoader with mixed precision support.
    """
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Move images and targets to the device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items() if k != "image_path"} for t in targets]

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses.item()

            # Debugging: Print loss details per batch
            print(f"Epoch:{epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}")
            for loss_name, loss_value in loss_dict.items():
                print(f" {loss_name}:{loss_value.item():.4f} ", end='|')
            print("")

        print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss:.4f}")
