import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from tqdm import tqdm


class VGG16Model(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16Model, self).__init__()
        # Load pre-trained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)

        # Freeze feature extractor layers
        for param in self.vgg16.features.parameters():
            param.requires_grad = False

        # Modify the classifier to match the number of classes
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vgg16(x)


# Define patch generation function
def split_into_patches(img, patch_size):
    """Split an image tensor into non-overlapping patches."""
    _, h, w = img.shape # C, H, W
    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(-1, 3, patch_size, patch_size) # Flatten patches
    return patches


# Define PatchDataset
class PatchDataset(Dataset):
    def __init__(self, dataset, patch_size):
        self.dataset = dataset
        self.patch_size = patch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        patches = split_into_patches(image, self.patch_size)
        return patches, label


# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for patches, labels in tqdm(dataloader, desc="Training"):
        # Average patch features
        batch_size, num_patches, channels, height, width = patches.size()
        patches = patches.view(batch_size * num_patches, channels, height, width).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(patches)
        outputs = outputs.view(batch_size, num_patches, -1).mean(dim=1) # Aggregate patch features
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for patches, labels in tqdm(dataloader, desc="Evaluating"):
            # Average patch features
            batch_size, num_patches, channels, height, width = patches.size()
            patches = patches.view(batch_size * num_patches, channels, height, width).to(device)
            labels = labels.to(device)

            outputs = model(patches)
            outputs = outputs.view(batch_size, num_patches, -1).mean(dim=1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


# Main training script
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    patch_size = 56
    num_classes = 10
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_dataset = PatchDataset(
        datasets.ImageFolder(root=r'C:\VSCode\KAN_satellite\Classification\EuroSAT_Dataset\train', transform=train_transform),
        patch_size=patch_size,
    )
    test_dataset = PatchDataset(
        datasets.ImageFolder(root=r'C:\VSCode\KAN_satellite\Classification\EuroSAT_Dataset\test', transform=test_transform),
        patch_size=patch_size,
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, loss, and optimizer
    model = VGG16Model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_loader, criterion, optimizer, device)
        evaluate(model, test_loader, criterion, device)

    # Save the trained model
    torch.save(model.state_dict(), "vgg16_patch_classifier.pth")
    print("Model saved as 'vgg16_patch_classifier.pth'")
