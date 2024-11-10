import os
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms

class DIV2KDataset:
    def __init__(self, upscale_factor=4, train=True, root_dir='data'):
        self.upscale_factor = upscale_factor
        self.train = train
        self.hr_dir = os.path.join(root_dir, 'HR')
        self.lr_dir = os.path.join(root_dir, 'LR')

        # Load and validate image filenames
        self.image_filenames = os.listdir(self.hr_dir)
        self.valid_image_filenames = []  # To hold valid filenames
        print(f"Found {len(self.image_filenames)} high-resolution images.")

        for filename in self.image_filenames:
            try:
                # Validate by trying to open the image
                img = Image.open(os.path.join(self.hr_dir, filename)).convert('RGB')
                self.valid_image_filenames.append(filename)
            except (UnidentifiedImageError, FileNotFoundError):
                print(f"Error loading image {filename}: skipping.")

        print(f"Found {len(self.valid_image_filenames)} valid high-resolution images.")

        # Define transformation for LR images
        self.transform = transforms.Compose([
            transforms.Resize((self.upscale_factor * 32, self.upscale_factor * 32)),  # Adjust as needed
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        # Use valid filenames for both HR and LR
        filename = self.valid_image_filenames[idx]
        hr_path = os.path.join(self.hr_dir, filename)
        lr_path = os.path.join(self.lr_dir, filename)

        try:
            hr_image = Image.open(hr_path).convert('RGB')
            lr_image = Image.open(lr_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error loading image {filename}: {e}")
            return None, None

        # Apply transformations if successfully loaded
        if hr_image is not None and lr_image is not None:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
    
        return lr_image, hr_image

    def __len__(self):
        return len(self.valid_image_filenames)
