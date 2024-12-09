import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_predictions(model, dataloader, device, class_names, num_samples=5, output_dir=None):
    """
   
    Args:
        model: 
        dataloader: 
        device: 
        class_names: 
        num_samples: 
        output_dir: 
    """
    model.eval()
    cmap = plt.get_cmap("tab20")  
    num_classes = len(class_names)

    # Fetch data from dataloader
    images_shown = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if images_shown >= num_samples:
                break

            images, labels = images.to(device), labels.to(device)
            preds = torch.argmax(model(images)['out'], dim=1)

            for i in range(images.size(0)):  
                if images_shown >= num_samples:
                    break

                
                image = images[i].cpu().numpy().transpose(1, 2, 0)  
                label = labels[i].cpu().numpy()
                pred = preds[i].cpu().numpy()

                
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(image / image.max())  
                axs[0].set_title("Original Image")
                axs[0].axis("off")

                axs[1].imshow(label, cmap=cmap, vmin=0, vmax=num_classes - 1)
                axs[1].set_title("Ground Truth")
                axs[1].axis("off")

                axs[2].imshow(pred, cmap=cmap, vmin=0, vmax=num_classes - 1)
                axs[2].set_title("Model Prediction")
                axs[2].axis("off")

                
                if output_dir:
                    plt.savefig(f"{output_dir}/sample_{images_shown + 1}.png")
                else:
                    plt.show()

                plt.close(fig)
                images_shown += 1

