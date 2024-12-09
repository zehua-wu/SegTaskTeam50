import torch

def save_checkpoint(model, optimizer, epoch, file_path="checkpoint.pth"):
   
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}")

def load_checkpoint(model, optimizer, file_path="checkpoint.pth"):
   
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from {file_path}, resuming from epoch {epoch + 1}")
    return model, optimizer, epoch

