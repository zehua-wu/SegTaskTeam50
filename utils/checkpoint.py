import os
import torch

# def save_checkpoint(model, optimizer, epoch, file_path="best_model.pth"):
#     """
#     Save the model and optimizer states along with the current epoch to a checkpoint file.
#     """
#     checkpoint = {
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "epoch": epoch,
#     }
#     torch.save(checkpoint, file_path)
#     print(f"Checkpoint saved to {file_path}")

# def load_checkpoint(model, optimizer, file_path="best_model.pth", device="cuda"):
#     """
#     Load the model and optimizer states along with the epoch from a checkpoint file.
#     If the checkpoint file does not exist, it will return the model and optimizer as is.
#     """
#     if not os.path.exists(file_path):
#         print(f"Checkpoint file '{file_path}' not found. Starting from scratch.")
#         return model, optimizer, 0  # Start from epoch 0 if no checkpoint is found

#     # Load checkpoint
#     print(f"Loading checkpoint from '{file_path}'...")
#     checkpoint = torch.load(file_path, map_location=device)

#     # Load model and optimizer states
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     epoch = checkpoint["epoch"]
#     print(f"Checkpoint loaded. Resuming from epoch {epoch + 1}")

#     return model, optimizer, epoch + 1  # Resume from the next epoch




def save_checkpoint(model, optimizer, epoch, model_name, save_dir="checkpoints"):
    """
    Save the model, optimizer states, and epoch to a checkpoint file for a specific model.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        epoch: Current epoch.
        model_name: Name of the model (to uniquely identify files).
        save_dir: Directory where checkpoints are saved.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    file_path = os.path.join(save_dir, f"{model_name}_best.pth")
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}")

def load_checkpoint(model, optimizer, model_name, save_dir="checkpoints", device="cuda"):
    """
    Load the model, optimizer states, and epoch for a specific model.

    Args:
        model: The model to load states into.
        optimizer: The optimizer to load states into.
        model_name: Name of the model (used to find the correct file).
        save_dir: Directory containing checkpoints.
        device: Device to map the checkpoint tensors to.
    
    Returns:
        model, optimizer, starting_epoch
    """
    file_path = os.path.join(save_dir, f"{model_name}_best.pth")
    if not os.path.exists(file_path):
        print(f"Checkpoint file '{file_path}' not found. Starting from scratch.")
        return model, optimizer, 0  # Start from epoch 0

    print(f"Loading checkpoint from '{file_path}'...")
    checkpoint = torch.load(file_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]

    print(f"Checkpoint loaded successfully. Resuming from epoch {epoch + 1}")
    return model, optimizer, epoch + 1
