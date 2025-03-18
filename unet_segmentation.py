import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the U-Net architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5)
        x = self.conv1(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.conv4(torch.cat([x, x1], dim=1))

        return self.outc(x)

# Custom Dataset class
class ImageSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Data transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation function
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_masks.extend(masks.cpu().numpy().flatten())

    accuracy = accuracy_score(all_masks, all_preds)
    precision = precision_score(all_masks, all_preds)
    recall = recall_score(all_masks, all_preds)
    f1 = f1_score(all_masks, all_preds)

    return total_loss / len(val_loader), accuracy, precision, recall, f1

# Main training loop
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training parameters
    num_epochs = 50
    batch_size = 8

    # Load and split training data (1000 images)
    train_data_dir = "path_to_training_data"  # Update with your path
    normal_images = [os.path.join(train_data_dir, "normal", f) for f in os.listdir(os.path.join(train_data_dir, "normal"))]
    masked_images = [os.path.join(train_data_dir, "masked", f) for f in os.listdir(os.path.join(train_data_dir, "masked"))]
    
    # Combine and shuffle data
    all_images = normal_images + masked_images
    all_masks = [f.replace("normal", "masks") for f in normal_images] + [f.replace("masked", "masks") for f in masked_images]
    
    train_images, val_images, train_masks, val_masks = train_test_split(
        all_images, all_masks, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = ImageSegmentationDataset(train_images, train_masks, transform=transform)
    val_dataset = ImageSegmentationDataset(val_images, val_masks, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, accuracy, precision, recall, f1 = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    # Load the best model for evaluation on the full dataset
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate on full dataset in cycles
    full_dataset_dir = "path_to_full_dataset"  # Update with your path
    all_full_images = [os.path.join(full_dataset_dir, f) for f in os.listdir(full_dataset_dir)]
    
    # Process in cycles of 25%
    cycle_size = len(all_full_images) // 4
    final_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for i in range(4):
        start_idx = i * cycle_size
        end_idx = (i + 1) * cycle_size if i < 3 else len(all_full_images)
        
        cycle_images = all_full_images[start_idx:end_idx]
        cycle_masks = [f.replace("images", "masks") for f in cycle_images]
        
        cycle_dataset = ImageSegmentationDataset(cycle_images, cycle_masks, transform=transform)
        cycle_loader = DataLoader(cycle_dataset, batch_size=batch_size)
        
        _, accuracy, precision, recall, f1 = evaluate_model(model, cycle_loader, criterion, device)
        
        final_metrics['accuracy'].append(accuracy)
        final_metrics['precision'].append(precision)
        final_metrics['recall'].append(recall)
        final_metrics['f1'].append(f1)
        
        print(f"\nCycle {i+1} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    # Print final average metrics
    print("\nFinal Average Metrics:")
    for metric, values in final_metrics.items():
        print(f"{metric.capitalize()}: {np.mean(values):.4f}")

if __name__ == "__main__":
    main() 
