# SegFormer training
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from data.dataset import get_dataloader
from configs.config import CONFIG, CLASS_NAMES
from utils.training import train_one_epoch, validate_one_epoch
from utils.utils_logger import setup_logger
from utils.checkpoint import load_checkpoint, save_checkpoint
from models.deeplabv3 import create_deeplabv3_mobilenet
from models.segformer import create_wrapped_segformer
from utils.visualize import visualize_predictions

# Get data from DataLoader
train_loader = get_dataloader(
    CONFIG["train_images_dir"],
    CONFIG["train_annotations_dir"],
    CONFIG["batch_size"],
    CONFIG["H"],
    CONFIG["W"],
)
# train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
test_loader = get_dataloader(
    CONFIG["val_images_dir"],
    CONFIG["val_annotations_dir"],
    batch_size=CONFIG["batch_size"],
    H=CONFIG["H"],
    W=CONFIG["W"],
)

if __name__ == "__main__":  # this is to correctly handle relevant process
   
  
    MODEL_NAME = "segformer"  

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, criterion, and optimizer
    model = create_wrapped_segformer(num_classes=19).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    # Load checkpoint if resuming training
    start_epoch = 0
    best_val_miou = 0.0

    if CONFIG["resume_training"]:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, MODEL_NAME)

    # Setup logger
    logger = setup_logger(MODEL_NAME)

    # Training Loop
    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")

        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Train Loss: {train_loss:.4f}")

        # Validation
        val_loss, val_miou, val_accuracy = validate_one_epoch(
            model, test_loader, criterion, device, num_classes=19, logger=logger, class_names=CLASS_NAMES
        )
        logger.info(f"Validation Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")

        # Save Best Checkpoint
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            save_checkpoint(model, optimizer, epoch, MODEL_NAME)
            logger.info(f"New best model saved with Val mIoU: {val_miou:.4f}")

        print(f"Finished epoch {epoch + 1}\n")

    # Visualization
    if CONFIG["visualize"]:
        logger.info("Visualizing predictions on validation set...")
        visualize_predictions(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=CLASS_NAMES,
            num_samples=CONFIG["num_visualize_samples"],
            output_dir=f"outputs/{MODEL_NAME}_visualizations",
        )

