import cv2
import copy
import pathlib
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ssd import MatchPrior
import mobilenetv1_ssd_config as cfg

class CustomDataset(Dataset):
    def __init__(self, root, anno_name, dataset_type, img_size, target_transform=None):
        self.root = pathlib.Path(root)
        self.anno_name = anno_name
        self.dataset_type = dataset_type
        self.img_size = img_size
        self.target_transform = target_transform

        self.data, self.class_names, self.class_dict = self._read_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_info = self.data[idx]
        image = self._read_image(image_info['image_name'])

        boxes = image_info['boxes']
        labels = image_info['labels']

        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        boxes, labels = self.target_transform(boxes, labels)

        return image_info['image_name'], image, boxes, labels

    def get_example(self, idx):
        image_info = self.data[idx]
        image = self._read_image(image_info['image_name'])
        # image = cv2.resize(image, (self.img_size, self.img_size))
        # image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)

        boxes = image_info['boxes']
        labels = image_info['labels']
        # boxes, labels = self.target_transform(boxes, labels)

        return image_info['image_name'], image, boxes, labels

    def _read_data(self):
        annotation_file = f"{self.root}/{self.anno_name}.csv"
        annotations = pd.read_csv(annotation_file)
        class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}

        data = []
        for image_name, group in annotations.groupby("ImageID"):
            boxes = group.loc[:, ['XMin', 'YMin', 'XMax', 'YMax']].values.astype(np.float32)
            labels = np.array([class_dict[name] for name in group["ClassName"]], dtype='int64')
            data.append({
                'image_name': image_name,
                'boxes': boxes,
                'labels': labels
            })
        return data, class_names, class_dict

    def _read_image(self, image_name):
        image_path = self.root / self.dataset_type / f"{image_name}.jpg"
        image = cv2.imread(str(image_path))
        if image.shape[-1] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

if __name__=='__main__':
    cd = CustomDataset('E:/DB_WIDERFACE/WIDER_train',
                       'train',
                       'images',
                       img_size=300,
                       target_transform=MatchPrior(cfg.priors,
                                                   cfg.center_variance,
                                                   cfg.size_variance, iou_threshold=0.5))
    #
    # cd = CustomDataset('C:/Users/th_k9/Desktop/git_pytorch-ssd-master/data',
    #                    'sub-train-annotations-bbox',
    #                    'train',
    #                    img_size=300,
    #                    target_transform=MatchPrior(cfg.priors,
    #                                                cfg.center_variance,
    #                                                cfg.size_variance, iou_threshold=0.5))

    name, image, box, label = cd.get_example(10)

    boxes = np.array(box)
    img = image

    h, w, _ = image.shape

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        cv2.rectangle(img, (int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), (255, 255, 0), 4)

    cv2.imshow('annotated', img)
    cv2.waitKey()
    cv2.destroyAllWindows()