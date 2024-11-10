import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError  # Import UnidentifiedImageError
import numpy as np

# Hyperparameters
num_epochs = 10
batch_size = 2
learning_rate = 0.001
log_interval = 10

# Define the dataset class
class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.hr_filenames = [f for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.lr_filenames = [f.replace('HR', 'LR') for f in self.hr_filenames]  # Assuming LR filenames match HR

        print(f"Found {len(self.hr_filenames)} high-resolution images.")

    def __len__(self):
        return len(self.hr_filenames)

    def __getitem__(self, idx):
        hr_filename = self.hr_filenames[idx]
        lr_filename = self.lr_filenames[idx]

        try:
            hr_image = Image.open(os.path.join(self.hr_dir, hr_filename)).convert('RGB')
            lr_image = Image.open(os.path.join(self.lr_dir, lr_filename)).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error loading image {os.path.join(self.hr_dir, hr_filename)}: {e}")


        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image

# Filter out None values from the dataset
def collate_fn(batch):
    # Remove None values
    batch = [item for item in batch if item[0] is not None and item[1] is not None]
    return tuple(zip(*batch)) if batch else ([], [])

# Define the model (Simple example, replace with your own model)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        return x

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))  # Resize images as needed
])

# Initialize dataset and DataLoader
hr_directory = "data/HR"  # Adjust path
lr_directory = "data/LR"  # Adjust path
train_dataset = DIV2KDataset(hr_directory, lr_directory, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Initialize the model, loss function, and optimizer
model = SimpleModel()  # Replace with your actual model
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()  # Or nn.L1Loss()

# Training Loop
for epoch in range(num_epochs):
    for i, (lr, hr) in enumerate(train_loader):
        if len(lr) == 0 or len(hr) == 0:
            continue  # Skip empty batches

        optimizer.zero_grad()  # Clear previous gradients

        # Forward pass
        outputs = model(torch.stack(lr))  # Model processes low-resolution images
        
        # Calculate loss
        loss = loss_function(outputs, torch.stack(hr))  # Ensure hr is also a tensor
        
        # Backward pass
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights
        
        if (i + 1) % log_interval == 0:  # Log the training status
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Training complete!")
