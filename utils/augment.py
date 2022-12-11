import albumentations as A
from albumentations.pytorch import ToTensorV2

def augmentation(phrase):
    if phrase == 'train':
        return A.Compose([
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.ColorJitter(),
            A.Normalize(),
            ToTensorV2()
        ])
    
    if phrase == 'test':
        return A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])