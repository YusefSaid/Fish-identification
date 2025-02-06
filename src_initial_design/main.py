import os
import sys
from datetime import datetime
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch.optim as optim


import constants
from data.dataset_fish_siamese import FishSiameseDataset
from data.dataset_fish_mask import FishMaskDataset
from models.siamese_network import SiameseNetwork
from transforms.custom_transforms import CustomTransform
from training.train import train_snn, train_mask_cnn
from training.evaluate import evaluate_snn
from utils.loss_utils import ContrastiveLoss
from utils.rois import extract_rois, generate_pairs_from_rois, generate_pairs_for_snn
from utils.collate_fn import collate_fn
from utils.visualize import visualize_predictions
from utils.model_io import save_model, load_model


if __name__ == "__main__":

    start_time = datetime.now()
    print(f'Start time: {start_time:%Y-%m-%d %H:%M:%S}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    # Apply custom transforms for preprocessing
    transform = CustomTransform(image_size=constants.IMAGE_SIZE)

    print('Loading dataset (masks) ...')
    train_loader_mask, test_loader_mask, class_labels = FishMaskDataset.create_dataloaders(
        image_base_folder=constants.IMAGE_BASE_FOLDER,
        mask_base_folder=constants.MASK_BASE_FOLDER,
        transform=transform,
        batch_size=constants.BATCH_SIZE,
        test_size=constants.TEST_SIZE,
        random_seed=None,
        collate_fn=collate_fn,
        max_images_per_class=constants.MAX_IMAGES_PER_CLASS
    )

    num_classes = len(class_labels)
    # # Last inn pre-trained Mask R-CNN
    model_rcnn = maskrcnn_resnet50_fpn(
        weights=None,
        num_classes=num_classes
    ).to(device)

    # Optimizer
    optimizer_cnn = optim.Adam(model_rcnn.parameters(), lr=constants.RCNN_LEARNING_RATE)

    print('Start training mask-r-cnn ...')
    train_mask_cnn(
        model=model_rcnn,
        train_loader=train_loader_mask,
        optimizer=optimizer_cnn,
        num_epochs=constants.NUM_EPOCHS_MRCNN,
        device=device,
        scaler=scaler
    )

    if constants.SAVE_MODELS:
        model_name = f'{start_time:%Y%m%d_%H%M%S}_rcnn'
        print(f"Saves Mask-r-cnn model ...({model_name})")
        save_model(model_rcnn, model_name)

    if constants.VISUALIZE_PREDICTIONS:
        print("Visualize predictions ...")
        visualize_predictions(
            model_rcnn,
            test_loader_mask,
            device,
            confidence_threshold=constants.RCNN_CONFIDENCE_THRESHOLD,
            class_labels_input=class_labels
        )

    # Generate pairs from ROIs
    print("Generating training and tests pairs from ROIs for Siamese Network ...")
    pairs, pair_labels, pairs_tests, pair_labels_tests = generate_pairs_for_snn(
        model_rcnn,
        train_loader_mask,
        test_loader_mask,
        device
    )

    # Create a custom DataLoader for Siamese training (based on mask-r-cnn network)
    print("Creating training and test DataLoaders for Siamese Network ...")
    train_loader, test_loader = FishSiameseDataset.create_dataloaders_from_mask_r_cnn(
        pairs=pairs,
        pair_labels=pair_labels,
        pairs_test= pairs_tests,
        pairs_labels_test=pair_labels_tests,
        transform=transform,
        batch_size=constants.BATCH_SIZE,
        shuffle=True
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
        num_epochs=constants.NUM_EPOCHS_SNN,
        device=device,
        scaler=snn_scaler
    )

    if constants.SAVE_MODELS:
        model_name = f'{start_time:%Y%m%d_%H%M%S}_snn'
        print(f"Saves SNN model ...({model_name})")
        save_model(model_rcnn, model_name)

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

