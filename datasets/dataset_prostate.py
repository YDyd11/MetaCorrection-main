# -*- encoding: utf-8 -*-
#Time        :2020/12/19 21:18:08
#Author      :Chen
#FileName    :polyp_dataset.py
#Version     :2.0

import os
import torch
import os.path as osp
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#from .transform import *
from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageFilter

def randomRotation(image, label):

    random_angle = np.random.randint(1, 60)
    return image.rotate(random_angle, Image.BICUBIC), label.rotate(random_angle, Image.NEAREST)

def randomColor(image):

    random_factor = np.random.randint(0, 31) / 10.  
    color_image = ImageEnhance.Color(image).enhance(random_factor)  
    random_factor = np.random.randint(10, 21) / 10.  
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  
    return ImageEnhance.Sharpness(brightness_image).enhance(random_factor)  

# Prostate Dataset
class Prostate(Dataset):
    def __init__(self, root, data_dir, mode='train', is_mirror=True, is_pseudo=None, max_iter=None):
        self.root = root
        self.data_dir = data_dir
        self.is_pseudo = is_pseudo
        self.is_mirror = is_mirror
        self.mode = mode
        self.imglist = []
        self.gtlist = []

        self.img_ids = [i_id.strip() for i_id in open(self.data_dir)]
        if not max_iter == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iter) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:

            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "image": img_file,
                "label": label_file,
                "name": name
            })
        #print(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        img_path = self.files[index]["image"]
        gt_path = self.files[index]["label"]
        name = datafiles["name"]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((256, 256), Image.BICUBIC)
        gt = Image.open(gt_path).convert('L')
        gt = gt.resize((256, 256), Image.NEAREST)
        if self.mode == 'train':
            #img = randomColor(img)
            img, gt = randomRotation(img, gt)
            #img = randomGaussian(img)

            if self.is_mirror:
                flip = np.random.choice(2) * 2 - 1
                img = img[:, :, ::flip]
                gt = gt[:, ::flip]

        img = np.asarray(img, np.float32)
        img = img / 255
        gt = np.asarray(gt, np.float32)
        gt = gt / 255
        # gt = torch.from_numpy(gt).long()

        img = img[:, :, ::-1]  # change to BGR
        img = img.transpose((2, 0, 1))
        size = img.shape

        data = {'image': img.copy(), 'label': gt.copy()}
        # if self.transform:
        #     data = self.transform(data)
        return img.copy(), gt.copy(), np.array(size), name

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    Source_data = Prostate(root='/data12T/ydaugust/data/MRI_Prostate/Source/', data_dir='/data12T/ydaugust/code/domain_adaptation/MetaCorrection-main/datasets/decathlon_list/train.txt', mode='train', max_iter=15000)
    print(Source_data.__len__())
    # for i in range(15000):
    #     print(Source_data[i])
    train_loader = torch.utils.data.DataLoader(Source_data, batch_size=8, shuffle=True, num_workers=4)
    print(np.max(Source_data[0][0]['image']), np.min(Source_data[0][0]['image']))