import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from datasets.gta5_dataset import GTA5DataSet
from datasets.decathlon_dataset import DecathlonDataSet
from torch.utils import data
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

class nciisbiDataSet(data.Dataset):
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
            img_file = osp.join(self.root, "%s" % (name))
            self.files.append({
                "image": img_file,
                "name": name
            })
            print(name)
        # print(self.files)
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["image"]).convert('RGB')
        print(datafiles["image"])
        name = datafiles["name"]
        # resize
        image = image.resize((256,256), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image/255
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        return image.copy(), np.array(size), name

class nciisbiPseudo(data.Dataset):
    def __init__(self, root, data_dir, mode='train', is_mirror=True, is_pseudo=True, max_iter=None):
        self.root = root
        self.data_dir = data_dir
        self.is_pseudo = is_pseudo
        self.is_mirror = is_mirror
        self.mode = mode
        self.imglist = []
        self.gtlist = []

        self.img_ids = [i_id.strip().split() for i_id in open(self.data_dir)]
        if not max_iter == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iter) / len(self.img_ids)))
        self.files = []
        for item in self.img_ids:
            image_path, label_path = item
            name = osp.splitext(osp.basename(label_path))[0]
            img_file = osp.join(self.root, image_path)
            label_file = osp.join(self.root, label_path)
            self.files.append({
                "image": img_file,
                "label": label_file,
                "name": name
            })
        #print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]["image"]
        gt_path = self.files[index]["label"]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((256,256), Image.BICUBIC)
        gt = Image.open(gt_path).convert('L')
        gt = gt.resize((256,256), Image.NEAREST)
        print(index)
        if self.mode == 'train':
            # img = randomColor(img)
            img, gt = randomRotation(img, gt)
            # img = randomGaussian(img)
            if self.is_mirror:
                flip = np.random.choice(2) * 2 - 1
                img = img[:, :, ::flip]
                gt = gt[:, ::flip]
        img = np.asarray(img, np.float32)
        img = img / 255
        gt = np.asarray(gt, np.float32)
        gt = gt / 255
        size = img.shape
        # gt = torch.from_numpy(gt).long()
        img = img[:, :, ::-1]  # change to BGR
        img = img.transpose((2, 0, 1))
        data = {'image': img.copy(), 'label': gt.copy()}
        # if self.transform:
        #     data = self.transform(data)
        return img.copy(), gt.copy(), np.array(size),  gt_path



if __name__ == '__main__':
    dst = nciisbiDataSet(root='/data12T/ydaugust/data/MRI_Prostate/Target/val', data_dir='/data12T/ydaugust/code/domain_adaptation/MetaCorrection-main/datasets/nci_list/val.txt', mode='train', max_iter=15000)
    trainloader = data.DataLoader(dst, batch_size=4)
    print(dst.__len__())
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
