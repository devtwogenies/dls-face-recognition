import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms

class CelebADataset(Dataset):
    def __init__(self, root_dir, landmark_file, img_dir, transform=None, sigma=2, heatmap_size=(64, 64)):
        """
        CelebA Dataset with landmark to heatmap conversion
        
        Args:
            root_dir (str): Root directory of the dataset
            landmark_file (str): Path to the landmark annotations file
            img_dir (str): Directory containing the images
            transform (callable, optional): Optional transform to be applied on images
            sigma (float): Standard deviation for Gaussian heatmap
            heatmap_size (tuple): Size of the output heatmap (height, width)
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, img_dir)
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        
        # Load landmarks
        self.landmarks = pd.read_csv(os.path.join(root_dir, landmark_file), 
                                   delim_whitespace=True, 
                                   header=None)
        
        # Define landmark column names
        self.landmark_cols = ['lefteye_x', 'lefteye_y', 
                            'righteye_x', 'righteye_y',
                            'nose_x', 'nose_y',
                            'leftmouth_x', 'leftmouth_y',
                            'rightmouth_x', 'rightmouth_y']
        
        # Set column names
        self.landmarks.columns = self.landmark_cols
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def create_heatmap(self, landmark, img_size):
        """
        Create a single heatmap for one landmark
        
        Args:
            landmark (tuple): (x, y) coordinates of the landmark
            img_size (tuple): Original image size (height, width)
            
        Returns:
            np.ndarray: Heatmap for the landmark
        """
        x, y = landmark
        h, w = img_size
        
        # Scale coordinates to heatmap size
        x = x * self.heatmap_size[1] / w
        y = y * self.heatmap_size[0] / h
        
        # Create coordinate grid
        y_grid, x_grid = np.mgrid[0:self.heatmap_size[0], 0:self.heatmap_size[1]]
        
        # Calculate distance from landmark
        dist = (x_grid - x) ** 2 + (y_grid - y) ** 2
        
        # Create Gaussian heatmap
        heatmap = np.exp(-dist / (2 * self.sigma ** 2))
        
        return heatmap

    def landmarks_to_heatmaps(self, landmarks, img_size):
        """
        Convert landmarks to heatmaps
        
        Args:
            landmarks (pd.Series): Landmark coordinates
            img_size (tuple): Original image size (height, width)
            
        Returns:
            np.ndarray: Stack of heatmaps
        """
        heatmaps = []
        
        # Process each landmark
        for i in range(0, len(landmarks), 2):
            x, y = landmarks[i], landmarks[i + 1]
            heatmap = self.create_heatmap((x, y), img_size)
            heatmaps.append(heatmap)
        
        return np.stack(heatmaps)

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path
        img_name = f"{idx:06d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        img_size = image.size[::-1]  # (height, width)
        
        if self.transform:
            image = self.transform(image)
        
        # Get landmarks
        landmarks = self.landmarks.iloc[idx].values
        
        # Convert landmarks to heatmaps
        heatmaps = self.landmarks_to_heatmaps(landmarks, img_size)
        heatmaps = torch.from_numpy(heatmaps).float()
        
        return {
            'image': image,
            'heatmaps': heatmaps,
            'landmarks': torch.from_numpy(landmarks).float()
        }

def get_celeba_dataset(root_dir, split='train', transform=None):
    """
    Helper function to create CelebA dataset
    
    Args:
        root_dir (str): Root directory of the dataset
        split (str): Dataset split ('train', 'val', 'test')
        transform (callable, optional): Optional transform to be applied on images
        
    Returns:
        CelebADataset: Dataset instance
    """
    # Define paths
    landmark_file = 'list_landmarks_align_celeba.txt'
    img_dir = 'img_align_celeba'
    
    # Create dataset
    dataset = CelebADataset(
        root_dir=root_dir,
        landmark_file=landmark_file,
        img_dir=img_dir,
        transform=transform
    )
    
    return dataset 