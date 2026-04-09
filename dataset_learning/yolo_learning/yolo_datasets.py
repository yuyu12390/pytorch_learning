import os
import torch
import xmltodict
from PIL import Image
from jedi.inference.compiled.access import object_class_dict
from torch.utils.data import Dataset
import torchvision


class YOLODataset(Dataset):
    def __init__(self,image_folder,label_folder,transform,label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.image_names = os.listdir(self.image_folder)
        self.classes_list = ["motor","no helmet","number","with helmet"]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.image_folder,img_name)
        img = Image.open(img_path).convert("RGB")
        #png -> xml
        label_name = img_name.split(".")[0] + ".txt"
        label_path = os.path.join(self.label_folder,label_name)
        with open(label_path, "r",encoding="utf-8") as f:
            label_content = f.read()
        object_infos = label_content.strip().split("\n")     #/n分割，去除头尾空字符串
        target = []
        for obj_info in object_infos:
            info_list = obj_info.strip().split(" ")
            class_id = float(info_list[0])
            center_x = float(info_list[1])
            center_y = float(info_list[2])
            width    = float(info_list[3])
            height   = float(info_list[4])
            target.extend([class_id,center_x,center_y,width,height])
        # label_dict = xmltodict.parse(label_content)
        # objects = label_dict["annotation"]["object"]
        # target = []
        # for obj in objects:
        #     object_name = obj["name"]
        #     object_class_id = self.classes_list.index(object_name)
        #     object_xmax = float(obj["bndbox"]["xmax"])
        #     object_ymax = float(obj["bndbox"]["ymax"])
        #     object_xmin = float(obj["bndbox"]["xmin"])
        #     object_ymin = float(obj["bndbox"]["ymin"])
        #     target.extend([object_class_id,object_xmin,object_ymin,object_xmax,object_ymax])
        target = torch.Tensor(target)
        if self.transform is not None:
            img = self.transform(img)

        return img,target


if __name__ == "__main__":
        train_dataset = YOLODataset(r"D:\yolo_datasets\HelmetDataset-YOLO-Train\images",r"D:\yolo_datasets\HelmetDataset-YOLO-Train\labels", transform=torchvision.transforms.ToTensor(),label_transform=None)
        print(len(train_dataset))
        print(train_dataset[11])
