import os
import torch

def save_checkpoint(model, optimizer, epoch, file_path="best_model.pth"):
    """
    Save the model and optimizer states along with the current epoch to a checkpoint file.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}")

def load_checkpoint(model, optimizer, file_path="best_model.pth", device="cuda"):
    """
    Load the model and optimizer states along with the epoch from a checkpoint file.
    If the checkpoint file does not exist, it will return the model and optimizer as is.
    """
    if not os.path.exists(file_path):
        print(f"Checkpoint file '{file_path}' not found. Starting from scratch.")
        return model, optimizer, 0  # Start from epoch 0 if no checkpoint is found

    # Load checkpoint
    print(f"Loading checkpoint from '{file_path}'...")
    checkpoint = torch.load(file_path, map_location=device)

    # Load model and optimizer states
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded. Resuming from epoch {epoch + 1}")

    return model, optimizer, epoch + 1  # Resume from the next epoch
