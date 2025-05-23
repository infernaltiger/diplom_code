from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from config import SQUARE_SIZE,dpi, width_inch, height_inch, supported_extensions


class DocumentDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['normal', 'flipped']
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        image = self.transform(image)

        return image, label

transform_train = transforms.Compose([
    transforms.Resize((SQUARE_SIZE, SQUARE_SIZE)),
    transforms.RandomRotation(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #стандартные значения
])
transform_val = transforms.Compose([
    transforms.Resize((SQUARE_SIZE, SQUARE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #стандартные значения
])

