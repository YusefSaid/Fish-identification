from datetime import datetime
import torch.optim as optim
import constants
from data.dataset_fish_siamese import FishSiameseDataset
from models.siamese_network import *
from transforms.custom_transforms import CustomTransform
from training.train import train_snn
from training.evaluate import evaluate_snn
from utils.loss_utils import ContrastiveLoss
from utils.model_io import  save_model


if __name__ == "__main__":

    start_time = datetime.now()
    print(f'Start time: {start_time:%Y-%m-%d %H:%M:%S}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    # Apply custom transforms for preprocessing
    transform = CustomTransform(image_size=constants.IMAGE_SIZE)

    # Create a custom DataLoader for Siamese training (based on mask-r-cnn network)
    print("Creating training and test DataLoaders for Siamese Network ...")
    print(f"Batch size: {constants.BATCH_SIZE}, Test size: {constants.TEST_SIZE}, "
          f"Pair count: {constants.PAIR_COUNT}, Distance threshold: {constants.DISTANCE_THRESHOLD}")
    train_loader, test_loader = FishSiameseDataset.create_dataloaders(
        constants.IMAGE_BASE_FOLDER,
        transform=transform,
        batch_size=constants.BATCH_SIZE,
        test_size=constants.TEST_SIZE,
        random_seed=42,  # Set to None for non-reproducibility
        pair_count=constants.PAIR_COUNT
    )

    # Initialize the Siamese Network
    print('Initialize Siamese Neural Network (SNN)...')
    model = SiameseNetwork().to(device)

    # Optimizer
    optimizer_snn = optim.Adam(model.parameters(), lr=constants.SNN_LEARNING_RATE)

    # Use contrastive loss for training
    criterion = ContrastiveLoss(margin=1.0)

    snn_scaler = torch.cuda.amp.GradScaler()
    print("Training SNN model...")
    train_snn(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer_snn,
        criterion=criterion,
        num_epochs=constants.NUM_EPOCHS_SNN,
        device=device,
        scaler=snn_scaler
    )

    if constants.SAVE_MODELS:
        model_name = f'{start_time:%Y%m%d_%H%M%S}_snn'
        print(f"Saves SNN model ...({model_name})")
        save_model(model, model_name)

    print("Evaluating SNN model...")
    avg_loss, accuracy = evaluate_snn(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    # Print evaluation results
    print(f"Average loss on test set: {avg_loss:.4f}")
    print(f"Accuracy on test set: {accuracy:.2f}%")

    print(f'End time: {datetime.now():%Y-%m-%d %H:%M:%S}')
    print(f'Time elapsed: {datetime.now() - start_time}')

