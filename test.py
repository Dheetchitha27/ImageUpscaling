import torch
from torch.utils.data import DataLoader
from dataset import DIV2KDataset
from srcnn import SRCNN
from utils import show_results

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = SRCNN().to(device)
model.load_state_dict(torch.load('srcnn.pth'))
model.eval()

# Test dataset and DataLoader
upscale_factor = 4
test_dataset = DIV2KDataset(upscale_factor=upscale_factor, train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Testing the model
with torch.no_grad():
    for idx, (lr, hr) in enumerate(test_loader):
        lr, hr = lr.to(device), hr.to(device)
        
        # Perform super-resolution
        sr = model(lr)
        
        # Display results
        show_results(lr.cpu(), sr.cpu(), hr.cpu())
