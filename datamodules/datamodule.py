import sys
sys.path.append('')
import cv2
import glob
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from utils.augment import augmentation

class CustomDataset(Dataset):
    def __init__(self, path, phrase, augment):
        super(CustomDataset, self).__init__()
        self.path= path
        self.augment= augment
        self.image_list= glob.glob(path + f'/{phrase}/*/*.jpg')
        self.label_list= [int(x.split('/')[-2]) for x in self.image_list]
        
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image= self.image_list[index]
        image= cv2.imread(image)
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augment is not None:
            image= self.augment(image= image)['image']
            
        label= self.label_list[index]
        
        return image, label
    
class DataModule(LightningDataModule):
    def __init__(self, path):
        super(DataModule, self).__init__()
        self.save_hyperparameters(logger= False)
        # self.path= path
        self.train_ds= CustomDataset(path, 'train', augmentation('train'))
        self.val_ds= CustomDataset(path, 'val', augmentation('test'))
    
    def train_dataloader(self):
        return self._loader(self.train_ds, is_train= True)
    
    def val_dataloader(self):
        return self._loader(self.val_ds, is_train= False)
    
    def _loader(self, dataset, is_train):
        return DataLoader(
            dataset= dataset,
            batch_size= 1,
            shuffle= is_train,
            num_workers= torch.cuda.device_count() * 4 if torch.cuda.is_available() else 0,
            pin_memory= True
        )
        
if __name__ == '__main__':
    path= 'TinyImageNet'
    dm= DataModule(path)
    train_dl= dm.train_dataloader()
    for batch, (img, label) in enumerate(train_dl):
        print('img: ', img)
        print('label: ', label)
        break
    
