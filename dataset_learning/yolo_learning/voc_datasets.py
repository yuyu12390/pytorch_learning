import os

import torch
import xmltodict
from PIL import Image
from jedi.inference.compiled.access import object_class_dict
from torch.utils.data import Dataset
import torchvision


class VOCDataset(Dataset):
    def __init__(self,image_folder,label_folder,transform,label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.image_names = os.listdir(self.image_folder)
        self.classes_list = ["no helmet","motor","number","with helmet"]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.image_folder,img_name)
        img = Image.open(img_path).convert("RGB")
        #png -> xml
        label_name = img_name.split(".")[0] + ".xml"
        label_path = os.path.join(self.label_folder,label_name)
        with open(label_path, "r",encoding="utf-8") as f:
            label_content = f.read()
        label_dict = xmltodict.parse(label_content)
        objects = label_dict["annotation"]["object"]
        target = []
        for obj in objects:
            object_name = obj["name"]
            object_class_id = self.classes_list.index(object_name)
            object_xmax = float(obj["bndbox"]["xmax"])
            object_ymax = float(obj["bndbox"]["ymax"])
            object_xmin = float(obj["bndbox"]["xmin"])
            object_ymin = float(obj["bndbox"]["ymin"])
            target.extend([object_class_id,object_xmin,object_ymin,object_xmax,object_ymax])
        target = torch.Tensor(target)
        if self.transform is not None:
            img = self.transform(img)

        return img,target


if __name__ == "__main__":
        train_dataset = VOCDataset(r"D:\HelmetDataset\train\images",r"D:\HelmetDataset\train\labels", transform=torchvision.transforms.ToTensor(),label_transform=None)
        print(len(train_dataset))
        print(train_dataset[11])
