# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from models.deeplab import MyDeepLab
from torchvision import transforms
from data.dataset import get_dataloader
from configs.config import CONFIG
from utils.training import train_one_epoch, validate_one_epoch
from utils.utils_logger import setup_logger
import torchvision


# Get data from DataLoader
train_loader = get_dataloader(CONFIG["train_images_dir"], CONFIG["train_annotations_dir"], CONFIG["batch_size"])
#train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
test_loader = get_dataloader(CONFIG["val_images_dir"],CONFIG["val_annotations_dir"], batch_size=CONFIG["batch_size"])


# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = MyDeepLab(num_classes=CONFIG["num_classes"]).to(device)
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
model.classifier[4] = nn.Conv2d(256, 12, kernel_size=(1, 1))  # 修改为 12 个类别

# 检查辅助分类器是否存在
if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
    model.aux_classifier[4] = nn.Conv2d(256, 12, kernel_size=(1, 1))  # 修改为 12 个类别

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 
                    lr=CONFIG["learning_rate"], 
                    momentum=CONFIG["momentum"], 
                    weight_decay=CONFIG["weight_decay"], 
                    nesterov=CONFIG["nesterov"])


# Training Loop

num_epochs = CONFIG["num_epochs"]
# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#     correct = 0
#     total = 0

#     for images, masks in train_loader:
#         images, masks = images.to(device), masks.to(device)

#         outputs = model(images)
#         loss = criterion(outputs, masks)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         correct += (predicted == masks).sum().item()
#         total += masks.numel()

#     train_accuracy = 100. * correct / total
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")


# for epoch in range(num_epochs):
#     print(f"Epoch {epoch + 1}/{num_epochs}")

#     # 训练一轮
#     train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

#     # 验证一轮并解包返回值
#     val_loss, val_iou, val_accuracy = validate_one_epoch(model, test_loader, criterion, device, num_classes=12)

#     # 打印结果
#     print(f"Epoch {epoch + 1}/{num_epochs}, "
#           f"Train Loss: {train_loss:.4f}, "
#           f"Val Loss: {val_loss:.4f}, "
#           f"Val IoU: {val_iou:.4f}, "
#           f"Val Pixel Accuracy: {val_accuracy:.4f}"
#           f"\n")




# 初始化 logger
logger = setup_logger(log_file="training_validation.log")

for epoch in range(num_epochs):
    logger.info(f"Epoch {epoch + 1}/{num_epochs}")

    # 训练一轮
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    logger.info(f"Train Loss: {train_loss:.4f}")

    # 验证一轮并解包返回值
    val_loss, val_iou, val_accuracy = validate_one_epoch(model, test_loader, criterion, device, num_classes=12, logger=logger)

    # 记录验证结果
    logger.info(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Val Loss: {val_loss:.4f}, "
        f"Val IoU: {val_iou:.4f}, "
        f"Val Pixel Accuracy: {val_accuracy:.4f}\n"
    )
