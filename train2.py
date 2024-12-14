# ViT training
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
    # Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # model = create_deeplabv3(num_classes=19, pretrained=True).to(device)
    # model = MyDeepLab(CONFIG["num_classes"])

    # model = create_deeplabv3_mobilenet(num_classes=19).to(device)


    model = create_wrapped_segformer(num_classes=19).to(device)


    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    
    # Suggested optimizer for transformer-based models
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    start_epoch = 0
    best_val_miou = float(0)

    if CONFIG["resume_training"]:
        model, optimizer, start_epoch = load_checkpoint(
            model, optimizer, file_path="checkpoint_dp.pth"
        )

    logger = setup_logger(log_file="training_validation_deeplab.log")



    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        logger.info(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")

        # Training
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Train Loss: {train_loss:.4f}")

        # Validation
        val_loss, val_miou, val_accuracy = validate_one_epoch(
            model,
            test_loader,
            criterion,
            device,
            num_classes=19,
            logger=logger,
            class_names=CLASS_NAMES,
        )
        logger.info(f"Validation Loss: {val_loss:.4f}")



        # Save into checkpoint

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            save_checkpoint(model, optimizer, epoch, file_path="best_model_deeplab.pth")
            logger.info(f"New best model saved with Val mIoU: {val_miou:.4f}")

        print()
        print(f"finsihing for epoch: {epoch+1}")

    if CONFIG["visualize"]:
        logger.info("Visualizing predictions on validation set...")
        visualize_predictions(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=CLASS_NAMES,
            num_samples=CONFIG["num_visualize_samples"],
            output_dir=CONFIG.get("visualize_output_dir", None),  # output direction
        )
