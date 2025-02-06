import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import constants

def visualize_predictions(model,
                          dataloader,
                          device,
                          confidence_threshold=constants.RCNN_CONFIDENCE_THRESHOLD,
                          class_labels_input: dict[str, int] =  None):

    try:
        # fix class labels (add background as 0)
        class_labels = {0 : 'background'}
        for k,v in class_labels_input.items():
            class_labels[v+1] = k
        print(class_labels)

        model.eval()
        fig, ax = plt.subplots(figsize=(10, 10))

        with torch.no_grad():
            for images, targets in dataloader:
                # Move images to device
                images = [img.to(device) for img in images]

                # Get predictions
                predictions = model(images)

                for idx, prediction in enumerate(predictions):
                    image = images[idx].cpu()  # Original image tensor
                    boxes = prediction["boxes"].cpu()  # Predicted bounding boxes
                    labels = prediction["labels"].cpu()  # Predicted labels
                    scores = prediction["scores"].cpu()  # Predicted confidence scores

                    # Convert the image back to a PIL image for visualization
                    image_pil = to_pil_image(image)

                    # Create a matplotlib figure for visualization
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    ax.imshow(image_pil)

                    # Draw the bounding boxes
                    for box, label, score in zip(boxes, labels, scores):
                        if score >= confidence_threshold:
                            xmin, ymin, xmax, ymax = box
                            rect = patches.Rectangle(
                                (xmin, ymin), xmax - xmin, ymax - ymin,
                                linewidth=2, edgecolor='r', facecolor='none'
                            )
                            ax.add_patch(rect)

                            # Add label and score
                            label_text = f"{class_labels[label.item()] if class_labels else label.item()} ({score:.2f})"
                            ax.text(
                                xmin, ymin - 5, label_text,
                                color='red', fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5)
                            )

                    plt.axis("off")
                    #plt.savefig(f"predictions_{idx}.png")
                    plt.show()
    except Exception as ex:
        print(f'Failed to visualize predictions: {ex}')