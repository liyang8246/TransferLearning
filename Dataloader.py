from torch_directml import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class pre_train(Dataset):
    def __init__(self,class_path) -> None:
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
        return (self.items[index][0],Image.open(self.items[index][1]))

    
if __name__ == "__main__":
    d = pre_train('data/pre_train')
    i,img = d.__getitem__(100)
    img.show()