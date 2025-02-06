import constants
from torch.cuda.amp import GradScaler, autocast

def train_snn(model,
              train_loader,
              optimizer,
              criterion, num_epochs, device, scaler):
    model.train()
    threshold = constants.DISTANCE_THRESHOLD  # Use a dynamic or preset threshold

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, ((images1, images2), labels) in enumerate(train_loader):
            # Move data to the device
            images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            with autocast():
                distances = model(images1, images2)
                loss = criterion(distances, labels.float())

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # Batch accuracy
            predictions = (distances < threshold).float()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Print debugging information for each batch
            print(f"\rEpoch:{epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)} "
                  f"Loss: {loss.item():.4f})", end=" ")


        # Epoch metrics
        epoch_accuracy = (correct_predictions / total_samples) * 100
        avg_loss = total_loss / len(train_loader)
        print(f"Summary: Epoch {epoch + 1}/{num_epochs}, Avg.Loss: {avg_loss:.4f}, "
              f"Accuracy: {epoch_accuracy:.2f}% (Distance Threshold: {threshold:.2f})")


