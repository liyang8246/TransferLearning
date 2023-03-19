from torch_directml import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class GetData(Dataset):
    def __init__(self,class_path,tf) -> None:
        self.tf = tf
        class_path = (class_path + '/') if class_path[-1] != "/" else class_path
        class_list = [class_path + i + '/' for i in os.listdir(class_path)]
        img_list = [os.listdir(class_list[i]) for i in range(len(class_list))]
        self.items = {}
        idx = 0
        for i,imgs in enumerate(img_list):
            for img in imgs:
                img_path = class_list[i] + img
                self.items.update({idx:(i,img_path)})
                idx += 1
    
    def __len__(self):
        return list(self.items.keys())[-1]

    def __getitem__(self, index):
        img = Image.open(self.items[index][1])
        img = self.tf(img)
        img = img[0].unsqueeze(dim=0)
        return (self.items[index][0],img)

    
if __name__ == "__main__":
    d = GetData('data/pre_train',tf = transforms.ToTensor())
    for i,img in d:
        print(img.shape)